import wikipediaapi
import json
import os
from datasets import Dataset

wikipedia = wikipediaapi.Wikipedia('Dataset (vifirsanova@gmail.com)', 'ru')

articles = '''Математика
Физика
Химия
Биология
Информатика
Искусственный интеллект
Машинное обучение
Обработка естественного языка
Глубокое обучение
Нейронная сеть
История России
Великая Отечественная война
Московский Кремль
Петр I
Екатерина II
Литература
Русская литература
Лев Толстой
Фёдор Достоевский
Антон Чехов
Музыка
Пётр Чайковский
Модест Мусоргский
Российская Федерация
География России
Сибирь
Байкал
Экономика России
Космонавтика
Юрий Гагарин
Мир
Европейский союз
Соединённые Штаты Америки
Китай
Япония
Культура
Русские обычаи
Русская кухня
Борщ
Щи
Искусство
Русская икона
Эрмитаж
Государственная Третьяковская галерея
Спорт
Олимпийские игры
Хоккей
Футбол
Владимир Путин
Александр Пушкин'''.split('\n')

with open('wiki_texts', "w", encoding="utf-8") as outfile:
    for article_title in articles:
        page = wikipedia.page(article_title)
        if page.exists():
            outfile.write(f"Title: {article_title}\n")
            outfile.write(page.text + "\n\n")
