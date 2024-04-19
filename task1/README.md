# Задание 1. BiGrams

## Структура:

* [data](./data/) — русские имена (мужские, женские) и русские фамилии (мужские) в формате `.txt`

* [dist](./dist/) — артефакты работы скриптов:

    * [firstnames](./dist/firstnames/) — артефакты обучения модели на именах

    * [lastnames](./dist/lastnames/) — артефакты обучения модели на фамилиях

* [test.py](./test.py) — скрипт теста модели

* [train.py](./train.py) — скрипт обучения модели

* [utils.py](./utils.py) — общие функции

## Запуск:

### Имена (firstnames)

* Для обучения модели на датасете имён необходимо ввести в консоли команду:

    ```bash
    python train.py --firstnames
    ```

* Примерный вывод консоли:

    ```bash
    Vocabulary (symbol to index) is saved on the path "/<path>/task1/dist/firstnames/stoi.pth"
    Vocabulary (index to symbol) is saved on the path "/<path>/task1/dist/firstnames/itos.pth"
    100%|██████████████████████████████████████████████| 30000/30000 [00:04<00:00, 6525.47it/s]
    Model is trained, loss: 2.00
    Trained model is saved on the path "/<path>/task1/dist/firstnames/model.pth"
    ```

* Для теста модели необходимо ввести в консоли команду:

    ```bash
    python test.py --firstnames
    ```

* Примерный вывод консоли:

    ```bash
    Test loss: 2.35

    Generated names:
    * Ила
    * Федославгус
    * Сольяк
    * Мамарлотия
    * Агарриан
    ```

**P.S.** Вместо параметра `--firstnames` можно передать один из следующих: `-f`, `--first`, `--names`.

### Фамилии (lastnames)

* Для обучения модели на датасете фамилий необходимо ввести в консоли команду:

    ```bash
    python train.py --lastnames
    ```

* Примерный вывод консоли:

    ```bash
    Vocabulary (symbol to index) is saved on the path "/<path>/task1/dist/lastnames/stoi.pth"
    Vocabulary (index to symbol) is saved on the path "/<path>/task1/dist/lastnames/itos.pth"
    100%|█████████████████████████████████████████████| 30000/30000 [00:04<00:00, 6093.34it/s]
    Model is trained, loss: 1.99
    Trained model is saved on the path "/<path>/task1/dist/lastnames/model.pth"
    ```

* Для теста модели необходимо ввести в консоли команду:

    ```bash
    python test.py --lastnames
    ```

* Примерный вывод консоли:

    ```bash
    Test loss: 2.02

    Generated names:
    * Конденекоровейсоврин
    * Марчов
    * Нутин
    * Весттынин
    * Балов
    ```


**P.S.** Вместо параметра `--lastnames` можно передать один из следующих: `-l`, `--last`, `--surnames`.
