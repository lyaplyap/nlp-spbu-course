# Задание 2. GPT

## Структура:

* [data](./data/) — роман Ильи Ильфа и Евгения Петрова "Двенадцать стульев" в формате `.txt`

* [dist](./dist/) — артефакты работы моделей:

    * [Default Model](./dist/default_model/) — артефакты стандартной модели (`GPT`)

    * [Funnel Transformer](./dist/funnel_transformer/) — артефакты изменённой модели (`Funnel Transformer`)

* [gpt.py](./gpt.py) — скрипт с решением задания

* [perplexity.png](./perplexity.png) — график со значениями perplexity исходной модели и изменённой модели

## Вывод работы программы:

* GPT:

    ```txt
    10.837376 M parameters
    step 0: train loss 4.9861, val loss 4.9806
    step 500: train loss 2.2360, val loss 2.2539
    step 1000: train loss 1.5420, val loss 1.7029
    step 1500: train loss 1.3288, val loss 1.5919
    step 2000: train loss 1.1155, val loss 1.6112
    step 2500: train loss 0.9437, val loss 1.6271
    step 3000: train loss 0.7949, val loss 1.7248
    step 3500: train loss 0.6620, val loss 1.7948
    step 4000: train loss 0.5036, val loss 1.8494
    step 4500: train loss 0.4242, val loss 1.9864
    step 4999: train loss 0.3392, val loss 2.1299

    — Тимофа? — один стул вдруг внезапнулся Воробьянинов. — Почему у вас их настоящий?

    — А может, вы скоро вашу есть знаменительно.

    — Как же будете?

    — Да того быть, Ипполит Матвеевич берегите наш ответственный разнокотченник против посетителя пассажирских рабочей и электричества «тангодушев».

    В этот Старопионе проделало все время. Доход явились трамвае и принялся ел, нового мужчинуть!

    — Узнаймет! — кричал он.

    — Отнец, восемь! — ужеливо сказал дворник, но отставительно засмедовольный в гобель.
    ```

* Funnel Transformer:

    ```txt
    10.837376 M parameters
    step 0: train loss 4.8717, val loss 4.8711
    step 500: train loss 2.0832, val loss 2.1100
    step 1000: train loss 1.5472, val loss 1.7045
    step 1500: train loss 1.3011, val loss 1.6153
    step 2000: train loss 1.1037, val loss 1.6010
    step 2500: train loss 0.9271, val loss 1.6227
    step 3000: train loss 0.7657, val loss 1.7280
    step 3500: train loss 0.6152, val loss 1.8332
    step 4000: train loss 0.4821, val loss 1.9932
    step 4500: train loss 0.3911, val loss 2.0316
    step 4999: train loss 0.3042, val loss 2.1711

    — Где это же не расспрашивал, может быть? Обудет, а вы случае не знаете? Это насчет! Вланы пишите?

    — Су часть перекупщика. А скотира величайших в полных пошлов проратнов пожарнее. Жирнов не видит человек, мелчише от мирабы дирижевый бюсток.

    — Ну?

    — Разя, вы знаете, что на это ваших работы… неприятно?.:

    — Вы всюду были любители?

    На Водой ущукинскинский движутитель, и большой горизной знали в дане адресе есть минуту в обществе. Привык котало того грянулось человека.

    — Да, — возмутил Остап, —
    ```
