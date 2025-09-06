# agent_setup.py

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent  # updated import

from langchain.memory import ConversationBufferMemory
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from db_connection import get_db_connection


def setup_agent_executor():
    """
    Sets up a dynamic SQL agent with memory and error handling.
    Returns the agent executor.
    """

    # 1. Initialize LLM and embeddings
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 2. Connect to database
    db = get_db_connection()

    # 3. Few-shot examples (helps LLM understand SQL generation)
    examples = [
        {"input": "List all employees in the Sales department.",
         "query": "SELECT * FROM employees WHERE department = 'Sales';"},
        {"input": "What is the average salary of engineers?",
         "query": "SELECT AVG(salary) FROM employees WHERE department = 'Engineering';"},
        {"input": "Show the top 3 highest paid employees.",
         "query": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3;"},
    ]

    # 4. Semantic example selector
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,
        Chroma,
        k=2,
        input_keys=["input"]
    )

    # 5. Memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 6. SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # 7. Create the SQL agent (dynamic, works for any question)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,  # crucial for output parsing issues
    )

    return agent_executor


if __name__ == "__main__":
    agent_executor = setup_agent_executor()

    # Example questions
    questions = [
        "How many employees are there in total?",
        "Who is the highest paid employee?",
        "Give me the name of the employee with the second highest salary.",
        "List all employees in the Engineering department.",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        response = agent_executor.invoke({"input": q})
        print("A:", response["output"])


# # agent_setup.py

# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.agents import create_sql_agent
# from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# from langchain.prompts import (
#     ChatPromptTemplate,
#     FewShotPromptTemplate,
#     MessagesPlaceholder,
#     PromptTemplate,
#     SystemMessagePromptTemplate,
# )
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

# from db_connection import get_db_connection


# def setup_agent_executor():
#     """
#     Sets up the advanced SQL agent with RAG, memory, and custom prompts.
#     Returns the agent executor.
#     """
#     # 1. Initialize LLM and Embeddings
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # 2. Get DB connection
#     db = get_db_connection()

#     # 3. Few-Shot Examples
#     examples = [
#         {
#             "input": "List all employees in the 'Sales' department.",
#             "query": "SELECT * FROM employees WHERE department = 'Sales';",
#         },
#         {
#             "input": "What is the average salary of engineers?",
#             "query": "SELECT AVG(salary) FROM employees WHERE department = 'Engineering';",
#         },
#         {
#             "input": "Show me the top 3 highest paid employees.",
#             "query": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3;"
#         },
#     ]

#     # 4. Semantic Example Selector
#     example_selector = SemanticSimilarityExampleSelector.from_examples(
#         examples,
#         embeddings,
#         Chroma,
#         k=2,
#         input_keys=["input"],
#     )

#     # 5. System Instructions
#     system_prefix = """You are an agent designed to interact with a MySQL database.
#     Given an input question, create a syntactically correct MySQL query to run, then look at the results of the query and return the answer.
#     Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
#     You can order the results by a relevant column to return the most interesting examples.
#     Do not make any DML statements (INSERT, UPDATE, DELETE, DROP).
#     If data is not in the database, say you do not know.
#     Always rephrase answers in a conversational tone.
#     """

#     few_shot_prompt = FewShotPromptTemplate(
#         example_selector=example_selector,
#         example_prompt=PromptTemplate.from_template(
#             "User input: {input}\nSQL query: {query}"
#         ),
#         input_variables=["input"],
#         prefix=system_prefix,
#         suffix="",
#     )

#     # 6. Prompt with memory (✅ added chat_history placeholder)
#     full_prompt = ChatPromptTemplate.from_messages(
#         [
#             SystemMessagePromptTemplate(prompt=few_shot_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),   # 👈 FIX
#             ("human", "{input}"),
#             MessagesPlaceholder(variable_name="agent_scratchpad"),
#         ]
#     )

#     # 7. Memory (✅ memory_key matches prompt)
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",   # 👈 FIX
#         return_messages=True
#     )

#     # 8. Toolkit
#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)

#     # 9. Create SQL Agent
#     agent_executor = create_sql_agent(
#         llm=llm,
#         toolkit=toolkit,
#         prompt=full_prompt,
#         agent_type="openai-tools",
#         verbose=True,
#         memory=memory,               # 👈 FIX
#         handle_parsing_errors=True,
#     )

#     return agent_executor


# if __name__ == '__main__':
#     agent_executor = setup_agent_executor()

#     print("--- Testing Agent ---")
#     response = agent_executor.invoke({"input": "How many employees are there in total?"})
#     print("Q: How many employees are there in total?")
#     print("A:", response["output"])

#     response = agent_executor.invoke({"input": "Who is the highest paid employee?"})
#     print("\nQ: Who is the highest paid employee?")
#     print("A:", response["output"])
