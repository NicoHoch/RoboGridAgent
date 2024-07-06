import os
from dotenv import load_dotenv

from services.robo import RoboService

load_dotenv()
langchainKey = os.getenv("LANGCHAIN_API_KEY")
openaiKey = os.getenv("OPENAI_API_KEY")

roboService = RoboService()

agent_finished = False

while not agent_finished:
    base64_image: str = roboService.get_image()

    next_move = roboService.get_next_move(image=base64_image)

    roboService.execute_move(next_move)

    base64_image: str = roboService.get_image()

    agent_finished = roboService.is_finished(image=base64_image)
