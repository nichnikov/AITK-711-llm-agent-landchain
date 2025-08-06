# This file contains the logic for creating and assembling the main LangChain processing chain.

# src/chains/processing_chain.py
import yaml
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.parsers.output_parsers import DocumentIdParser, SentenceExtractionParser, FinalAnswerParser

def load_prompts_from_config(config_path="config/prompts.yaml"):
    """Загружает шаблоны промтов из YAML файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_processing_chain(llm: ChatOpenAI, all_docs: list[dict]):
    """
    Создает и возвращает полную цепочку обработки запроса в соответствии с ТЗ.
    """
    prompts_config = load_prompts_from_config()
    
    # -- Шаг 1: Выбор релевантных документов --
    prompt1 = PromptTemplate.from_template(prompts_config['prompt1_selection']['template'])
    chain1 = prompt1 | llm | DocumentIdParser()

    # -- Шаг 2: Извлечение предложений из выбранных документов --
    def filter_docs_by_ids(input_data: dict) -> str:
        """Фильтрует исходные документы по ID, полученным от первого шага."""
        selected_ids = input_data['selected_ids']
        if isinstance(selected_ids, str): # Обработка случая "нет информации"
            return selected_ids
        
        docs_map = {doc['doc_id']: doc for doc in all_docs}
        
        # Собираем контекст для второго промта
        context_for_prompt2 = []
        for doc_id in selected_ids:
            if doc_id in docs_map:
                doc = docs_map[doc_id]
                context_for_prompt2.append(
                    f"Из документа {doc['doc_id']} (заголовок: {doc['title']}):\n{doc['paragraph']}\n"
                )
        return "\n".join(context_for_prompt2)

    prompt2 = PromptTemplate.from_template(prompts_config['prompt2_extraction']['template'])
    chain2 = (
        {
            "context": itemgetter("selected_ids") | RunnablePassthrough(filter_docs_by_ids),
            "question": itemgetter("question")
        } 
        | prompt2 
        | llm 
        | SentenceExtractionParser()
    )

    # -- Шаг 3: Синтез финального ответа --
    def format_sentences_for_synthesis(extracted_data: list | str) -> str:
        """Форматирует извлеченные предложения для финального промта."""
        if isinstance(extracted_data, str):
            return extracted_data
        return "\n".join([f"[{item['sentence']}, {item['doc_id']}]" for item in extracted_data])
    
    def get_source_ids(extracted_data: list | str) -> list:
        if isinstance(extracted_data, str):
            return []
        return sorted(list(set(item['doc_id'] for item in extracted_data)))

    prompt3 = PromptTemplate.from_template(prompts_config['prompt3_synthesis']['template'])
    chain3 = (
        {
            "context": itemgetter("extracted_sentences") | RunnablePassthrough(format_sentences_for_synthesis),
            "question": itemgetter("question"),
        }
        | prompt3
        | llm
        | FinalAnswerParser()
    )
    
    # -- Сборка полной цепочки (Master Chain) --
    # Используем LCEL для передачи данных между шагами
    full_chain = (
        {
            "selected_ids": chain1,
            "question": itemgetter("question"), # Пробрасываем исходный вопрос
        }
        | RunnablePassthrough.assign(extracted_sentences=chain2) # Выполняем chain2 и добавляем результат
        | chain3 # Выполняем chain3
    )
    
    return full_chain


if __name__ == "__main__":
    prmts = load_prompts_from_config()
    print(prmts)