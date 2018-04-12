import tensorflow as tf
from simple_network.src.utils import *

def multilayer_perceptron(input_tensor, weights, biases):
    """
    
    Описание нашей нейронной сети
    
    :param input_tensor: входные данные
    :param weights: вектор весов
    :param biases: 
    :return: 
    """

    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_addition)

    # Скрытый слой с RELU активацией
    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2_activation = tf.nn.relu(layer_2_addition)

    # Слой вывода с линейной активацией
    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition



if __name__ == '__main__':

    # Читаем текст
    newsgroups_train, newsgroups_test = read_the_texts()

    # Создаем словарик
    vocab = create_vocab(newsgroups_train, newsgroups_test)

    # Parameters
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 150
    display_step = 1

    # Параметры сети
    n_hidden_1 = 100  # количество признаков первого слоя
    n_hidden_2 = 100  # количество признаков второго слоя
    n_classes = 3  # Категории
    n_input = len(vocab)  # Слова в словаре

    # Запускаем процесс обучения #
    # Веса и смещения хранятся в переменных tf.Variable,
    # которые содержат состояние в графе между вызовами run(). В машинном обучении
    # принято работать с весами и смещениями, полученными через нормальное распределение:


    # Инициализируем длобальные переменные. Определяем входной и выходной плейсхолдеры

    input_tensor = tf.placeholder(tf.float32, [None, n_input], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, n_classes], name="output")

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Конструирование модели
    prediction = multilayer_perceptron(input_tensor, weights, biases)

    # Определение функции потерь
    entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
    loss = tf.reduce_mean(entropy_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Запуск графа
    with tf.Session() as sess:
        sess.run(init)

        # Тренировочный цикл
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(newsgroups_train.data) / batch_size)

            # Цикл по всем блокам - батчам
            for i in range(total_batch):
                batch_x, batch_y = get_batch(newsgroups_train, i, batch_size, vocab)
                # Запускаем операцию оптимизации (backprop) и вычисление стоимости (получение loss)
                c, _ = sess.run([loss, optimizer], feed_dict={input_tensor: batch_x, output_tensor: batch_y})
                # Вычисляем среднюю ошибку на этом батче
                avg_cost += c / total_batch
            # Выводим  ошибку на каждой эпохе
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=", \
                      "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        save_path = saver.save(sess, "./model/model.ckpt")
        print("Model saved in path: %s" % save_path)

        # Тестирование модели
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        # Считаем качество
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        total_test_data = len(newsgroups_test.target)
        batch_x_test, batch_y_test = get_batch(newsgroups_test, 0, total_test_data, vocab)
        print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))