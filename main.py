from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")

    information = """

            Pedro Sánchez Pérez-Castejón[n. 1] (Spanish pronunciation: [ˈpeðɾo ˈsantʃeθ ˈpeɾeθ kasteˈxon] ⓘ; born 29 February 1972)[1] is a Spanish politician and economist who has served as Prime Minister of Spain since 2018.[2][3] He has also been Secretary-General of the Spanish Socialist Workers' Party (PSOE) since July 2017, having previously held that office from 2014 to 2016, and has also been serving as the ninth president of the Socialist International since 2022.

            Sánchez began his political career in August 2004 as a city councillor in Madrid, before being elected to the Congress of Deputies in 2009. In 2014, he was elected Secretary-General of the PSOE, becoming Leader of the Opposition. He led the party through the inconclusive 2015 and 2016 general elections, but resigned as Secretary-General shortly after the latter, following public disagreements with the party's executive. He was re-elected in a leadership election eight months later, defeating internal rivals Susana Díaz and Patxi López.
                """
       
    summary_template = """
        given information {information} about a person I want you create:
        - a short summary of maximum 50 words
        - 2 interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
