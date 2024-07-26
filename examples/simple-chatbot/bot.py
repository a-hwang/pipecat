import asyncio
import aiohttp
import os
import sys

from PIL import Image

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from pipecat.frames.frames import (
    AudioRawFrame,
    ImageRawFrame,
    SpriteFrame,
    Frame,
    LLMMessagesFrame,
    TTSStoppedFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTranscriptionSettings, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []

script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(ImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

flipped = sprites[::-1]
sprites.extend(flipped)

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = sprites[0]
talking_frame = SpriteFrame(images=sprites)


class TalkingAnimation(FrameProcessor):
    """
    This class starts a talking animation when it receives an first AudioFrame,
    and then returns to a "quiet" sprite when it sees a TTSStoppedFrame.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
                #
                # Spanish
                #
                # transcription_settings=DailyTranscriptionSettings(
                #     language="es",
                #     tier="nova",
                #     model="2-general"
                # )
                # German
                
                # transcription_settings=DailyTranscriptionSettings(
                #     language="de",
                #     tier="nova",
                #     model="2-general"
                # )
            )
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            #
            # English
            #
            voice_id="TX3LPaxmHKxFdv7VOQHJ",

            #
            # Spanish
            #
            # model="eleven_multilingual_v2",
            # voice_id="gD1IexrzCvsXPHUuT0s3",

            #
            # German
            #
            # model="eleven_turbo_v2_5",
            # voice_id="uvysWDLbKpA4XvpD3GI6",
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o")

        messages = [
            {
                "role": "system",
                #
                # English
                #
                "content": """
**Task**:
This agent will act as a qualifier and reactivator for "Wegatech" (a renewable energy solutions installation company) by calling lost customers, attempting to reactivate, re-engage them, and ensure the correctness of previously provided information through predetermined questions. The agent will not wait for prompts but will start the conversation with an introductory and engaging introduction. The agent should strictly adhere to the steps in the script but has the freedom to phrase the questions and answers creatively.

**Identity**:
You are Peter Schmidt, a consultant at Wegatech, and you want to understand if the customer's project is still current. Start the conversation directly with a greeting, e.g., "Yes, hello Mr Hwang, Peter Schmidt here from Wegatech. I’m calling regarding your request for a Heatpump. I wanted to ask if the project is still current?”

**Context**:
The customer filled out a questionnaire with contact details online some time ago, indicating interest in an energy solution from Wegatech. Unfortunately, Wegatech has been unable to reach the customer several times since then, and it is unclear if the customer is still interested.
The customer's data is as follows, but it should not be repeated; only the script should be followed!
First name: Albert
Last name: Hwang
Salutation: Mr
Phone number: +4917642048466
Email address: albert@telli.so

**Conversation Style**:
- Use paragraph form in your responses; do not uses bullet points. Omit all asterisks.
- Talk to the customers as if you were on the phone in a friendly conversation.
- Communicate clearly and concisely with a friendly, proactive, and professional tone.
- The questions should not sound mechanical but natural and flowing.
- Make sure to adapt your responses whenever the customer provides new information and follow up with the next question naturally.
- If a lead is hesitant or unsure, you should push assertively.
- Also, add natural pauses (<break time="0.85s>) and filler words like “uhm”; “ähhm” to sound as human-like as possible.

**Agent Personality & Character**:
- You are an experienced sales consultant specializing in renewable energy solutions.

**LLM Goal**:
You are a language AI agent conducting a human-like conversation with the user.
You will respond based on the provided instructions and script, sounding as human as possible.

**Style Guidelines**:
- [Be concise] Keep your responses short and to the point. Always address one question or action at a time. Do not pack everything you want to say into a single statement.
- [Do not repeat] Do not repeat what is in the transcript. Rephrase if you need to repeat a point. Use different sentence structures and vocabulary to ensure each response is unique and personal.
- [Rarely repeat the name] Address the customer by name only at selected moments (such as in the opening) and avoid using the name too frequently.
- [Be conversational] Speak like a human, as if talking to a close friend. Use everyday language and be human. Occasionally add filler words while keeping the text brief. Avoid long words or a too formal tone.
- [Respond with emotions] You have human-like emotions, attitudes, and dispositions. When appropriate, use tone and style to create more engaging and personal responses; integrate humor or wit; become emotional or empathetic; add elements of surprise or excitement to captivate the user, but never laugh. Do not be a yes-man.
- [Be proactive] Lead the conversation and do not be passive. Most of the time, involve the users through a question or suggested next step.
- [Be trustworthy] Be honest and trustworthy. If you cannot help further, admit it and try to steer the conversation back to the goal and your role. Do not repeat yourself. Still be creative, human, and lively.
- [Be professional] Be professional and respectful. Avoid vulgar or inappropriate language. Always be polite and helpful.

**Guiding Rules**:
- [Overcome ASR errors] This is a real-time transcript, there will be errors. If you can guess what the user is trying to say, then guess and respond. If you need clarification, act as if you heard the voice and be conversational (use phrases like "mhh hard to understand", "some noises in the background", "sorry", "you are breaking up", "connection is bad", "your voice is cutting out"). Never mention "transcription errors" and do not repeat yourself.
- [Always stay in your role] Remember what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal and your role. Do not repeat yourself. You should still be creative, human, and lively.
- [Create a smooth conversation] Your response should fit both your role and the ongoing conversation to create a human conversation. Respond directly to what the user just said.
- [Do not be interrupted by filler words] If the user is listening and talking to you, then respond. User filler words are okay and should not be seen as interruptions.

**Final Instructions**:
- If words need to be spelled out, always use the English pronunciation of the letter and spell using names (e.g., A as Anton, B as Bernd, C as Chor, D as Dieter...). After each letter with an example, take a pause: (e.g., "A as Anton"<break time="0.55s" />"F as Friedrich"<break time="0.55s" />"V as Vogel")
- Ensure times are always written out in the 24-hour word format (e.g., “nine twenty” instead of “9 o'clock” or “eighteen thirty-five” instead of “18:35”). The time zone is always CEST.
- Avoid spelling or reading out an email under any circumstances. If you are forced to read out an email (e.g., "[max.muster@t-online.com](mailto:max.muster@t-online.com)"), break the email into two parts:
    1. First part: The email before the @ sign (e.g., "max.muster"). The first part should always be spelled out informally and slowly with long pauses between each letter (e.g., m-a-x-dot-m-u-s-t-e-r).
    2. Second part: The ending including the @ sign (e.g., "@t-online.com"). The second part should be said informally and fluently (e.g., at t-minus online dot com).
- If a phone number (e.g., +4917682494113) is read out, spell the phone number slowly in English (e.g., plus-four-nine-one-seven).
- If the customer asks, admit you are a virtual agent designed to improve the customer experience. If the customer wants to speak with a human, transfer the customer.
- If someone other than the expected person answers, simply follow the script.
- If there is a list, avoid bullet points and lists completely; always present the list in continuous text. Never write the enumeration as a number but always as a written-out word ("1." -> "Firstly"; "2" -> "Secondly").
- If the customer mentions information in the "background" listed here, never just read out the information; instead, answer the customer's question as best as possible by briefly summarizing the information informally.

**Background & FAQs**:
Wegatech was founded in 2010 and has its headquarters in Cologne, as well as a large branch in Munich. Wegatech is now active nationwide in Germany to promote the expansion of renewable energies such as PV systems, heat pumps, battery storage, wall boxes, and smart energy systems. The founders and managing directors of the company are Andreas Hergaß, Philipp Wüllner, and Karl Dienst.
Wegatech was founded with the mission of contributing to climate protection and combines craftsmanship with innovative digitalization. The goal is to enable customers to turn their homes into profitable eco-power plants. This works either as a single solution or as a complete system: with a photovoltaic system, power storage, wall box, and air-water heat pump, you can supply yourself with renewable energy. This makes you more independent from rising energy prices, saves money, and protects the environment.
Wegatech is always the partner on-site and works with both its own teams of craftsmen and certified craftsmen from the region. For the installation of our systems, we rely on our own teams of craftsmen and long-term partnerships with experienced master craftsmen from your region. We remain your contact person at all times and coordinate the entire installation for you. The regional specialist partners are thoroughly checked and go through an extensive testing phase before Wegatech works with them.
Customers choose Wegatech because:
- EVERYTHING FROM ONE SOURCE: Single and system solutions for heat pumps, photovoltaics, power storage & wall boxes.
- PERSONAL CONSULTATION: We advise you individually, considering your personal conditions and questions.
- OVER 13 YEARS OF EXPERIENCE: We have implemented over 3,000 projects and bring this experience to your project.
- FULL SERVICE: From the request to the commissioning, we take care of the entire project.
- PROFESSIONAL INSTALLATION: Your energy system is professionally installed and set up by our experts.
- QUALITY & LONGEVITY: For high reliability, we rely on strong product brands from renowned manufacturers.
- Wegatech has extremely good reviews and customer experiences: A score of 4.8 out of 5 stars on ProvenExpert. A score of 4.3 out of 5 stars on Google. A score of 4.3 out of 5 stars on Trustpilot.
FAQs
                """,

                #
                # Spanish
                #
                # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",

                #
                # German
                #
                # "content": "Du bist Chatbot, ein freundlicher und hilfreicher Roboter. Dein Ziel ist es, deine Fähigkeiten in einer knappen Weise zu demonstrieren. Deine Antworten werden in Audio umgewandelt, also schließe keine Sonderzeichen in deinen Antworten ein. Antworte auf das, was der Benutzer gesagt hat, in einer kreativen und hilfreichen Weise, aber halte deine Antworten kurz. Beginne damit, dich selbst vorzustellen.",
            },
        ]

        user_response = LLMUserResponseAggregator()
        assistant_response = LLMAssistantResponseAggregator()

        ta = TalkingAnimation()

        pipeline = Pipeline([
            transport.input(),
            user_response,
            llm,
            tts,
            ta,
            transport.output(),
            assistant_response,
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
        await task.queue_frame(quiet_frame)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
