# src/llm/client.py

import os
from langchain_openai import ChatOpenAI

def get_llm_client() -> ChatOpenAI:
    """
    Создает и настраивает LLM клиент.
    
    Приоритетно использует конфигурацию для LLM-агрегатора, если она задана
    в переменных окружения. В противном случае использует прямое подключение
    к OpenAI.
    
    Raises:
        ValueError: Если не найдена ни одна конфигурация (ни для агрегатора, ни для OpenAI).
        
    Returns:
        Экземпляр ChatOpenAI, настроенный на соответствующий эндпоинт.
    """
    aggregator_key = os.getenv("AGGREGATOR_API_KEY")
    aggregator_base_url = os.getenv("AGGREGATOR_API_BASE")
    
    openai_key = os.getenv("OPENAI_API_KEY")

    # Приоритет для агрегатора
    if aggregator_key and aggregator_base_url:
        print("--- Используется конфигурация LLM Агрегатора ---")
        return ChatOpenAI(
            model="gpt-4o", # Название модели может быть другим в вашем агрегаторе
            temperature=0,
            api_key=aggregator_key,
            base_url=aggregator_base_url,
        )
        
    # Запасной вариант - прямое подключение
    if openai_key:
        print("--- Используется прямое подключение к OpenAI API ---")
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_key
        )
        
    # Если никакие ключи не найдены
    raise ValueError(
        "Не удалось найти конфигурацию LLM. "
        "Пожалуйста, определите переменные AGGREGATOR_API_KEY и AGGREGATOR_API_BASE "
        "или OPENAI_API_KEY в вашем .env файле."
    )