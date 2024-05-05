#代码源自https://blog.csdn.net/aweizhenlihai/article/details/123073633
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
 
def read_csv_file(train_data_file_path, train_label_file_path, test_data_file_path, test_label_file_path):
    """
    读取csv文件并将文件进行拼接
    :param train_data_file_path: 训练数据路径
    :param train_label_file_path: 训练标签路径
    :param test_data_file_path: 测试数据路径
    :param test_label_file_path: 测试标签路径
    :return: 返回拼接完成后的路径
    """
    #从csv中读取数据
    train_data = pd.read_csv(train_data_file_path,header=None)
    train_label = pd.read_csv(train_label_file_path,header=None)
    test_data = pd.read_csv(test_data_file_path,header=None)
    test_label = pd.read_csv(test_label_file_path,header=None)
    ##########将数据集拼接起来
    #数据与标签拼接
    dataset_train = pd.concat([train_data,train_label],axis=1)
    dataset_test = pd.concat([test_data,test_label],axis=1)
    #训练集与测试集拼接
    dataset = pd.concat([dataset_train,dataset_test],axis=0).sample(frac=1,random_state=0).reset_index(drop=True)
    return dataset
 
def get_train_test(dataset, data_ndim = 1):
    """
    划分训练数据和测试数据，并转变数据维数
    :param dataset: 数据拼接
    :param data_ndim: 数据的维数
    :return: 训练集和测试集的标签和数据
    """
    #获得训练数据和标签
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
 
    #划分训练集和测试集
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
 
    #改变数据维度让他符合（数量，长度，维度）的要求
    X_train = np.array(X_train).reshape(X_train.shape[0],X_train.shape[1],data_ndim)
    X_test = np.array(X_test).reshape(X_test.shape[0],X_test.shape[1],data_ndim)
 
    print("X Train shape: ", X_train.shape)
    print("X Test shape: ", X_test.shape)
 
    return X_train, X_test, y_train, y_test
 
def bulid(X_train, y_train, X_test, y_test, model_path, batch_size=10, epochs=10):
    """
    搭建网络结构完成训练
    :param X_train: 训练集数据
    :param y_train: 训练集标签
    :param X_test: 测试集数据
    :param y_test: 测试集标签
    :param batch_size: 批次大小
    :param epochs: 循环轮数
    :return: acc和loss曲线
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.0001), input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.0001)),
        tf.keras.layers.Conv1D(filters=128, kernel_size=(3,), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.0001)),
        tf.keras.layers.MaxPool1D(pool_size=(3,), strides=2, padding='same'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.0001)),
        tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.0001)),
        tf.keras.layers.Dense(units=10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    model.save(model_path)
    model.summary()
    # 获得训练集和测试集的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # 绘制acc曲线
    #plt.subplot(1, 2, 1)
    #plt.plot(acc, label='Training Accuracy')
    #plt.plot(val_acc, label='Validation Accuracy')
    #plt.title('Training and Validation Accuracy')
    #plt.legend()
 
    # 绘制loss曲线
    #plt.subplot(1, 2, 2)
    #plt.plot(loss, label='Training Loss')
    #plt.plot(val_loss, label='Validation Loss')
    #plt.title('Training and Validation Loss')
    #plt.legend()
    #plt.show()
 
def main():
    x_test_csv_path = ".../lamost_test_data.csv"
    y_test_csv_path = ".../lamost_test_label.csv"
    x_train_csv_path = ".../lamost_train_data.csv"
    y_train_csv_path = ".../lamost_train_label.csv"
    model_path = ".../lamost_sp.keras"

    dataset = read_csv_file(x_train_csv_path, y_train_csv_path, x_test_csv_path, y_test_csv_path)
    X_train, X_test, y_train, y_test = get_train_test(dataset = dataset, data_ndim = 1)

    bulid(X_train, y_train, X_test, y_test, model_path)

if __name__ == "__main__":
    main()
