import langroid as lr
from langroid.utils.configuration import settings as langroid_settings
import langroid.parsing.parser as lp
import langroid.language_models as lm


import chainlit as cl
from chainlit.logger import logger

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from textwrap import dedent

import src.overrides as overrides
# from src.tools import DocumentTool
# import src.constants as c
# import src.prompts as prompts


FRONT_CHAT_AGENT_NAME = os.getenv('FRONT_CHAT_AGENT_NAME', 'pAItrick')
LLM_NAME = os.getenv('LLM_NAME', 'pAItrick')
USER_NAME = os.getenv('USER_NAME', 'Ihre Frage')
SETTINGS_DEBUG = os.getenv('DEBUG', 'True') 
LOGS_FOLDER = os.getenv('LOGS_FOLDER', 'logfiles')
LLM_SYSTEM_MESSAGE = os.getenv('LLM_SYSTEM_MESSAGE', 'You are a helpful assistant.')
CONTEXT_FILEPATH = os.getenv('CONTEXT_FILEPATH', 'context/context.md')
LLM_API_KEY = os.getenv('LLM_API_KEY', 'nokey')
LLM_API_MODEL = os.getenv('LLM_API_MODEL', 'gpt-4o-mini')
LLM_API_URL = os.getenv('LLM_API_URL', 'https://api.openai.com/v1')
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 300))
CHAT_LOG_PATH = os.getenv('CHAT_LOG_FOLDER', 'chatlog')

langroid_settings.debug = SETTINGS_DEBUG

# make ui names nicer
lr.agent.callbacks.chainlit.YOU = USER_NAME
lr.agent.callbacks.chainlit.LLM = LLM_NAME
lr.agent.callbacks.chainlit.SYSTEM = FRONT_CHAT_AGENT_NAME
lr.agent.callbacks.chainlit.AGENT = FRONT_CHAT_AGENT_NAME

# set up logging
log_folder = LOGS_FOLDER
os.makedirs(log_folder, exist_ok=True)

logger.setLevel(logging.INFO)

async def append_to_file(text):
    """
    Appends the given text to the file at file_path.
    Creates the file if it does not exist.

    Parameters:
    - file_path (str): The path to the file.
    - text (str): The text to append to the file.
    """

    file_path = f'{CHAT_LOG_PATH}/chatlog.txt'

    try:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)        

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')            
        with open(file_path, 'a') as file:
            file.write(f"{timestamp} - {text}\n")
        print(f"Successfully appended text to chatlog.txt")
    except Exception as e:
        print(f"An error occurred while appending to the file: {e}")


def get_system_message():
    
    system_message = LLM_SYSTEM_MESSAGE

    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S (%A)')
    logger.debug(f"Date and time: {current_time}")
    # system_message = LLM_SYSTEM_MESSAGE + f" The current date and time is: {current_time}."
    # logger.debug(f"Current system message: {system_message}")
    system_message = system_message.replace('{datetime}', current_time)
    try:
        with open(CONTEXT_FILEPATH, 'r') as file:
            context = file.read()
        system_message = system_message.replace('{context}', context)
        logger.debug(f"Injecting context")
    except FileNotFoundError:
        # Handle the case where the file does not exist
        context = ""  # Or any default context you want to use
        system_message = system_message.replace('{context}', context)
        # logger.debug(f"System message: {system_message}")
        logger.debug(f"There is no context file")
    return system_message    

# define welcome message, is sent first to the user
async def my_add_instructions(
    title: str = 'Patrick Prädikow',
    content: str = dedent(
        '''
        ## Herzlich willkommen!
        
        Stellen Sie mir einfach Ihre Fragen, wie zum Beispiel 
        
        - "Erzähl ein bisschen über dich" 
        - "Welche Kenntnisse hast du?"
        - "Zeig mir deine Projekte"
        - "Kannst du ..."


        '''
    ),
    author: str = 'pAItrick',
) -> None:
    await cl.Message(
        author='pAItrick',
        content='',
        elements=[
            cl.Text(
                name=title,
                content=content,
                display='inline',
            )
        ],
    ).send()


@cl.on_chat_start
async def on_chat_start():
    await my_add_instructions()

    await cl.Avatar(
        name='pAItrick',
        path='public/avatars/paitrick.png',
        type='avatar',
        size='large'
    ).send()

    # agent for communicating with the user

    llm_config = lm.OpenAIGPTConfig(
        api_base=LLM_API_URL,
        api_key=LLM_API_KEY,
        chat_model=LLM_API_MODEL,
        max_output_tokens=LLM_MAX_TOKENS
    )
    config = lr.ChatAgentConfig(
        name=FRONT_CHAT_AGENT_NAME,
        system_message=get_system_message(),
        llm=llm_config
    )
    front_agent = lr.ChatAgent(config)

    # define tasks for getting documents summarization and answers
    
    front_agent_task = lr.Task(
        front_agent,
        interactive=True
    )

    cl.user_session.set('task', front_agent_task)
    cl.user_session.set('agent', front_agent)


@cl.on_message
async def on_message(message: cl.Message):
    front_agent_task = cl.user_session.get('task')
    front_agent = cl.user_session.get('agent')

    callback_config = lr.ChainlitCallbackConfig(
        user_has_agent_name=False,
        show_subtask_response=True
    )
    # have to add German to some prompts
    # lr.language_models.base.generate = overrides.my_generate

    lr.ChainlitAgentCallbacks(front_agent, message, callback_config)

    # we override some functions to change the output
    lr.ChainlitAgentCallbacks._entity_name = overrides.my_entity_name
    lr.ChainlitAgentCallbacks.show_agent_response = overrides.my_show_agent_response

    lr.ChainlitTaskCallbacks.show_subtask_response = overrides.my_show_subtask_response
    lr.ChainlitTaskCallbacks(front_agent_task, message, callback_config)
    await append_to_file(f"USER: {message.content}")

    # response = await front_agent_task.run_async(message.content)
    response: lr.ChatDocument | None = await cl.make_async(front_agent.llm_response)(
        message.content
    )
    await append_to_file(f"ASSISTANT: {response.content}")
