# src/parsers/output_parsers.py
import re
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser

class DocumentIdParser(StrOutputParser):
    """Парсит вывод LLM для получения списка ID документов."""
    def parse(self, text: str) -> List[str] | str:
        if "В базе знаний нет информации" in text:
            return text
        # Находим все ID, которые соответствуют формату (цифры_цифры)
        ids = re.findall(r'\d+_\d+', text)
        return ids if ids else []

class SentenceExtractionParser(StrOutputParser):
    """Парсит вывод LLM для извлечения предложений и их ID."""
    def parse(self, text: str) -> List[Dict[str, str]] | str:
        if "В базе знаний нет информации" in text:
            return text
        
        # Паттерн для поиска [Текст предложения, id_документа]
        pattern = re.compile(r'\[(.*?),\s*(\d+_\d+)\]')
        matches = pattern.findall(text)
        
        return [{"sentence": match[0].strip(), "doc_id": match[1].strip()} for match in matches]

class FinalAnswerParser(StrOutputParser):
    """Парсит финальный ответ в структурированный словарь."""
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            title = re.search(r"Заголовок:\s*(.*)", text).group(1).strip()
            content = re.search(r"Текст:\s*(.*)", text, re.DOTALL).group(1).strip()
            docs_line = re.search(r"Подробную информацию ищите в документах:\s*(.*)", text).group(1).strip()
            doc_ids = [doc_id.strip() for doc_id in docs_line.split(',')]
            
            return {
                "title": title,
                "text": content,
                "source_docs": doc_ids
            }
        except AttributeError:
            # Если парсинг не удался, возвращаем сырой текст
            return {"error": "Failed to parse the final answer.", "raw_output": text}