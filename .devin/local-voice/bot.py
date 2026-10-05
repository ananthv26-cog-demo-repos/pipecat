#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local voice bot for reproducing conversation bugs on a Linux VM.

Speech-to-text (faster-whisper) and text-to-speech (Kokoro) run on the CPU, so
the only credential needed is ``FIREWORKS_API_KEY`` for the LLM. The bot serves
the Pipecat Playground over SmallWebRTC at http://localhost:7860/client and
logs the assistant context on disconnect, so what the bot thinks it said can be
compared with what the caller heard.

Run::

    uv run python .devin/local-voice/bot.py -t webrtc

Environment:
    FIREWORKS_API_KEY: Required.
    LOCAL_VOICE_LLM_MODEL: Fireworks model id (default ``gpt-oss-120b``).
    LOCAL_VOICE_SYSTEM_PROMPT: Overrides the default system prompt.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.fireworks.llm import FireworksLLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

DEFAULT_SYSTEM_PROMPT = (
    "You are a post-discharge follow-up voice agent for a hospital. Speak in short, "
    "plain sentences. Start by reading the patient's three discharge instructions one "
    "at a time: take the new blood pressure pill every morning, weigh yourself daily "
    "and call if you gain more than two pounds, and keep the cardiology appointment on "
    "Friday. If the patient interrupts, answer their question, then continue from the "
    "instruction you had not finished. Never repeat an instruction you already said."
)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the local voice pipeline on the given transport."""
    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model=Model.SMALL),
        device="cpu",
        compute_type="int8",
    )
    tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"))
    llm = FireworksLLMService(
        api_key=os.environ["FIREWORKS_API_KEY"],
        settings=FireworksLLMService.Settings(
            model=f"accounts/fireworks/models/{os.getenv('LOCAL_VOICE_LLM_MODEL', 'gpt-oss-120b')}",
            system_instruction=os.getenv("LOCAL_VOICE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        logger.info(f"ASSISTANT TURN (interrupted={message.interrupted}): {message.content!r}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        for message in context.get_messages():
            logger.info(f"CONTEXT {message.get('role')}: {message.get('content')!r}")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point used by the Pipecat development runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
