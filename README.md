# flask-backend
Backend for Yandex.Mood Mediaservices Hackathon Project

На самом деле еще и фронтенд...

## Установка
Проект работает на Python 3.7 и использует наработки следующих репозиториев:
* Yandex Music Api (неофициальная библиотека): [git](https://github.com/MarshalX/yandex-music-api)
* PyTorch Unsupervised Sentiment Discovery: [git](https://github.com/NVIDIA/sentiment-discovery)
* Music-Emotion-Recognition (с большими доработками): [git](https://github.com/danz1ka19/Music-Emotion-Recognition)

Для запуска нужно:
* установить пакеты из requirements.txt
* установить [pytorch версии 1.0.1](https://pytorch.org/get-started/previous-versions/#v101)
* положить в папку dl файл с классификатором: [ссылка](https://drive.google.com/file/d/1ieiWFrYBqzBgGPc3R36x9oL7vlj3lt2F/view)

## Описание

Решение применяет модели получения эмоций из текста и определения 
настроения музыки к истории прослушивания пользователя Яндекс.Музыки.

Результаты применения моделей отображаются в веб-интерфейсе на графиках.

В период с 25.08.2021 до 28.08.2021 решение запущено на сервере.

Чтобы получить его адрес, обратитесь в tg [@vldplcd](https://t.me/vldplcd), 
а то решение требовательное, задудосите ещё(((

