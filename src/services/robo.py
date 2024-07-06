import base64

from models.robo import IsFinished, Move

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class RoboService:
    def __init__(self, start_index=0) -> None:
        self.index: int = start_index
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0, max_tokens=500, max_retries=2
        )

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image(self):
        image_path = f"images/image{self.index}.png"
        return self.encode_image(image_path)

    def get_next_move(self, image: str) -> Move:

        systemMessage = SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are in control of a robot, and your primary objective is to navigate the robot to reach the flag in the fewest steps possible without touching fields with a Mountain! The terrain includes fields that you cannot access or cross, specifically those with mountains. You have four possible actions to direct the robot: move upwards, move downwards, move right, and move left. Directions are as seen by the observer. Additionally, you must specify the number of fields the robot should move in the chosen direction. You will be provided with a visual representation of the current state of the terrain, including the robot's position, the position of the Mountains and the location of the flag. Based on this information, you need to decide and provide the next move the robot should make. The format for your output should be [direction, steps], where 'direction' indicates the chosen action (up, down, right, left) and 'steps' represents the number of fields to move. For example, an output could be [forward, 5], meaning the robot should move forward five fields. Return only the very next move. Remember, the goal is to find the most efficient path to the flag, avoiding fields with mountains. Please avoid suggesting moves that cross or land on mountains.",
                }
            ]
        )

        humanMessage = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Return the next move.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ]
        )

        chat = [systemMessage, humanMessage]

        structured_llm = self.llm.with_structured_output(Move)

        response: Move = structured_llm.invoke(chat)
        return response

    def execute_move(self, move: Move):
        print(f"executing move: {move}")
        self.index += 1

    def is_finished(self, image: str) -> bool:
        chat = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are checking if the robot finished it's goal and is on the same field as the flag.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ]
        )

        structured_llm = self.llm.with_structured_output(IsFinished)

        response: IsFinished = structured_llm.invoke([chat]).finished
        return response
