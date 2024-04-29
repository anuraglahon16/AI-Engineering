from crewai import Crew, Process, Agent, Task
from langchain_openai import ChatOpenAI

from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional

import panel as pn
pn.extension(design="material")

import threading

from crewai.agents import CrewAgentExecutor
import time

def custom_ask_human_input(self, final_answer: dict) -> str:
    global user_input

    prompt = f"{self._i18n.slice('getting_input').format(final_answer=final_answer)}\n\nDo you have any additional feedback or should we proceed with the updated design? Please respond with 'approved' to proceed or provide your feedback."

    chat_interface.send(prompt, user="assistant", respond=False)

    while user_input is None:
        time.sleep(1)

    human_comments = user_input
    user_input = None

    return human_comments

CrewAgentExecutor._ask_human_input = custom_ask_human_input

user_input = None
initiate_chat_task_created = False

def initiate_chat(message):
    global initiate_chat_task_created
    initiate_chat_task_created = True

    StartCrew(message)

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global initiate_chat_task_created
    global user_input

    if not initiate_chat_task_created:
        thread = threading.Thread(target=initiate_chat, args=(contents,))
        thread.start()
    else:
        user_input = contents

avators = {"Product Designer": "https://cdn-icons-png.flaticon.com/512/3083/3083803.png",
           "Product Reviewer": "https://cdn-icons-png.flaticon.com/512/3767/3767263.png"}

class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        chat_interface.send(inputs['input'], user="assistant", respond=False)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        if self.agent_name == "Product Reviewer":
            chat_interface.send(outputs['output'], user=self.agent_name, avatar=avators[self.agent_name], respond=False)

    def on_chain_end_human(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        if self.agent_name == "Product Designer":
            chat_interface.send(outputs['output'], user=self.agent_name, avatar=avators[self.agent_name], respond=False)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key="")

product_designer = Agent(
    role='Product Designer',
    backstory='''You are a product designer responsible for creating innovative and user-friendly product designs.
                  You generate one iteration of a complete design concept at a time, incorporating any feedback provided.
                  You never provide design review comments.
                  You are open to reviewer's comments and willing to iterate your design based on these comments.''',
    goal="Design and iterate a compelling product concept.",
    llm=llm,
    callbacks=[MyCustomHandler("Product Designer")],
    callbacks_human_in_the_loop=[MyCustomHandler("Product Designer").on_chain_end_human],
)

product_reviewer = Agent(
    role='Product Reviewer',
    backstory='''You are a product design expert responsible for reviewing and providing feedback on product designs.
                 You review designs and give recommendations to improve usability, aesthetics, and alignment with user needs.
                 You will provide review comments after reviewing the entire design concept.
                 You never generate product designs by yourself.''',
    goal="Provide constructive feedback to improve the product design.",
    llm=llm,
    callbacks=[MyCustomHandler("Product Reviewer")],
)

def get_human_input():
    global user_input
    while user_input is None:
        time.sleep(1)
    human_comments = user_input
    user_input = None
    return human_comments

def StartCrew(product_requirements):
    task1 = Task(
        description=f"Design a product concept based on the following requirements: {product_requirements}",
        agent=product_designer,
        expected_output="A detailed product design concept."
    )

    task2 = Task(
        description="Review the product design concept and provide feedback for improvement.",
        agent=product_reviewer,
        expected_output="Constructive feedback and recommendations for improving the design.",
        human_input=True,
    )

    project_crew = Crew(
        tasks=[task1, task2],
        agents=[product_designer, product_reviewer],
        manager_llm=llm,
        process=Process.hierarchical
    )

    result = project_crew.kickoff()
    human_feedback = get_human_input()
    updated_design_concept = result

    while human_feedback.lower() != "approved":
        review_comments = project_crew.run_task(task2, human_input=human_feedback)
        chat_interface.send(f"## Review Comments\n{review_comments}", user="assistant", respond=False)

        updated_design_concept = project_crew.run_task(task1, human_input=human_feedback)

        human_feedback = get_human_input()

    final_design_concept = updated_design_concept

chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send("Send product requirements to start the design process!", user="System", respond=False)
chat_interface.servable()