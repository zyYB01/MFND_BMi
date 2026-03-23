from keras.src.layers import Multiply, MultiHeadAttention
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Attention
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, MaxPool1D, Concatenate, Reshape, \
    Dropout, LSTM, Bidirectional, Dense, Attention
from keras.layers import *
import keras
from keras.layers import Input, InputLayer, Concatenate, Reshape, Conv1D, LeakyReLU, BatchNormalization, MaxPool1D, \
    Bidirectional, LSTM, Dropout, Dense
from keras.models import Model, load_model, Sequential
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import keras.regularizers


def myloss(y_true,y_pred):
    y_pred_norm=tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1))
    y_pred_norm1=tf.expand_dims(y_pred_norm,-1)
    y_pred_norm2=tf.expand_dims(y_pred_norm,1)
    y_pred2=tf.transpose(y_pred,[0,2,1])
    y_pred_dot12=tf.matmul(y_pred,y_pred2)
    y_pred_norm12=tf.matmul(y_pred_norm1,y_pred_norm2)
    angle_mat=tf.math.divide(y_pred_dot12,y_pred_norm12)
    map=tf.constant([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    ,[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    ,[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    ,[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1]],dtype=float)
    angle_mat2=tf.multiply(angle_mat,map)
    angle_mat2=tf.reduce_sum(angle_mat2,axis=-1)
    weight=1.1-angle_mat2
    weight=tf.expand_dims(weight,1)
    error=tf.reduce_sum(tf.square(y_pred-y_true), axis=-1)
    error=tf.expand_dims(error,2)
    error=tf.matmul(weight,error)
    error=tf.reduce_sum(tf.square(error), axis=-1)

    print(error.shape)

    return error



class LSTMRegression(object):

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=1,batch_size=10):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size

    def fit(self,X_train,y_train,x_val,y_val):

        model=Sequential()
        model.add(Bidirectional(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=0,kernel_regularizer=keras.regularizers.l2(0.01),return_sequences=False)))

        if self.dropout!=0: model.add(Dropout(self.dropout))

        model.add(BatchNormalization())

        model.add(Dense(y_train.shape[1]))

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        model.compile(loss='mse',optimizer='rmsprop',metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])

        history=model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose,validation_data=(x_val,y_val),callbacks=[reduce_lr],batch_size=self.batch_size)
        self.model=model
        return history

    def predict(self,X_test):

        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted


class CNN_lstm_v2(object):
    def __init__(self,units=100,dropout=0,num_epochs=100,verbose=1,batch_size=128):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size

    def fit(self,X_train,y_train,x_val,y_val):

        model=Sequential()
        model.add(InputLayer(input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Conv1D(192,1,strides=1,padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(self.units,  dropout=self.dropout,
                                     recurrent_dropout=0, kernel_regularizer=keras.regularizers.l2(0.01),
                                     return_sequences=True)))
        model.add(Dropout(self.dropout)) 
        model.add(MaxPool1D(16, strides=4))
        model.add(Conv1D(2,1,strides=1,padding="same"))                       

        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss='mse', optimizer=optimizer,
                      metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])

        checkpoint = ModelCheckpoint(filepath="result/best_model/best_model.hdf5", monitor='val_loss',verbose=1,save_best_only='True',mode='min')

        history = model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose,
                                validation_data=(x_val, y_val),callbacks=[reduce_lr,checkpoint], batch_size=self.batch_size)
        self.model = load_model("result/best_model/best_model.hdf5",custom_objects={'myloss':myloss})
        return history

    def predict(self,X_test):

        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted

class CNN_lstm_v2_lfp(object):
    def __init__(self,units=100,dropout=0,num_epochs=100,verbose=1,batch_size=128):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size

    def fit(self,X_train,y_train,x_val,y_val):

        model=Sequential()
        model.add(InputLayer(input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Conv1D(192,1,strides=1,padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(MaxPool1D(2,strides=1))
        model.add(Bidirectional(LSTM(self.units,  dropout=self.dropout,
                                     recurrent_dropout=0, kernel_regularizer=keras.regularizers.l2(0.01),
                                     return_sequences=False)))
        model.add(Dropout(self.dropout))
        model.add(Dense(y_train.shape[1]))
        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss='mse', optimizer=optimizer,
                      metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])

        checkpoint = ModelCheckpoint(filepath="result/temp/best_model/best_model.keras", monitor='val_loss',verbose=1,save_best_only='True',mode='min')

        history = model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose,
                                validation_data=(x_val, y_val),callbacks=[reduce_lr,checkpoint], batch_size=self.batch_size)
        self.model = load_model("result/temp/best_model/best_model.keras")
        return history

    def predict(self,X_test):

        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.activations import softmax

class CNN_lstm_v2_BH(object):
    def __init__(self, units=100, dropout=0, num_epochs=100, verbose=1, batch_size=128, num_heads=8):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_heads = num_heads

    def fit(self, X_train, y_train, x_val, y_val):
        inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = Conv1D(80, 1, strides=1, padding="same")(inputs)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Conv1D(80, 1, strides=1, padding="same")(inputs)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=64)(x, x)
        x = MaxPool1D(2, strides=1)(x)
        x = Bidirectional(LSTM(self.units, dropout=self.dropout,
                               recurrent_dropout=0, kernel_regularizer=l2(0.01),
                               return_sequences=False))(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath="result/temp/best_model/BH_best_model.keras", monitor='val_loss',
                                     verbose=1, save_best_only='True', mode='min')

        history = model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose,
                            validation_data=(x_val, y_val), callbacks=[reduce_lr, checkpoint],
                            batch_size=self.batch_size)
        self.model = load_model("result/temp/best_model/best_model.keras")
        return history

    def predict(self, X_test):
        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted


class CNN_lstm_v2_lfp_f(object):
    def __init__(self,units=150,dropout=0.5,num_epochs=50,verbose=1,batch_size=128, model_path="result/temp/best_model/best_model1.keras"):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size
         self.model_path = model_path

    def fit(self,X_train,y_train,x_val,y_val):

        model=Sequential()
        model.add(InputLayer(input_shape=(X_train.shape[1],)))
        model.add(Reshape((-1, X_train.shape[1])))
        model.add(Conv1D(192,1,strides=1,padding="same"))
        model.add(LeakyReLU())
        model.add(Conv1D(192, 1, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(MaxPool1D(1,strides=1))
        model.add(Bidirectional(LSTM(self.units,  dropout=self.dropout,
                                     recurrent_dropout=0, kernel_regularizer=keras.regularizers.l2(0.01),
                                     return_sequences=False)))
        model.add(Dropout(self.dropout))

        model.add(Dense(y_train.shape[1]))

        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss='mse', optimizer=optimizer,
                      metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])

        checkpoint = ModelCheckpoint(filepath=self.model_path, monitor='val_loss',verbose=1,save_best_only='True',mode='min')

        history = model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose,
                                validation_data=(x_val, y_val),callbacks=[reduce_lr,checkpoint], batch_size=self.batch_size)
        self.model = load_model(self.model_path)
        return history

    def predict(self,X_test):

        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted


class CNN_lstm_v2_lfp_ff(object):
    def __init__(self, units=100, dropout=0, num_epochs=100, verbose=1, batch_size=128):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size

    def fit(self, X_train1, X_train2, y_train, x_val1, x_val2, y_val):
        input1 = Input(shape=(X_train1.shape[1],))
        input2 = Input(shape=(X_train2.shape[1],))

        processed1 = Reshape((-1, X_train1.shape[1]))(input1)
        processed1 = Conv1D(192, 3, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(192, 3, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(192, 4, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(192, 4, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)

        processed2 = Reshape((-1, X_train2.shape[1]))(input2)
        processed2 = Conv1D(192, 1, strides=1, padding="same")(processed2)
        processed2 = BatchNormalization()(processed2)
        processed2 = LeakyReLU()(processed2)

        merged = Concatenate()([processed1, processed2])
        merged = Conv1D(192, 1, strides=1, padding="same")(merged)
        merged = BatchNormalization()(merged)
        merged = LeakyReLU()(merged)
        merged = MaxPool1D(1, strides=1)(merged)

        lstm_out = Bidirectional(LSTM(self.units, dropout=self.dropout,
                                      recurrent_dropout=0, kernel_regularizer=keras.regularizers.l2(0.01),
                                      return_sequences=False))(merged)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(self.dropout)(lstm_out)
        predictions = Dense(y_train.shape[1])(lstm_out)

        model = Model(inputs=[input1, input2], outputs=predictions)

        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        checkpoint = ModelCheckpoint(filepath="result/temp/best_model/best_model.keras", monitor='val_loss', verbose=1, save_best_only='True', mode='min')

        model.compile(loss='mse', optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])

        history = model.fit([X_train1, X_train2], y_train, epochs=self.num_epochs, verbose=self.verbose,
                            validation_data=([x_val1, x_val2], y_val), callbacks=[reduce_lr, checkpoint], batch_size=self.batch_size)

        self.model = load_model("result/temp/best_model/best_model.keras")

        return history

    def predict(self, X_test1, X_test2):
        y_test_predicted = self.model.predict([X_test1, X_test2])
        return y_test_predicted



class CNN_LSTM_SelfAttention(object):
    def __init__(self, units=100, dropout=0, num_epochs=100, verbose=1, batch_size=128):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size

    def self_attention(self, input_tensor, index):
        # Self-Attention Mechanism
        attention_probs = Dense(input_tensor.shape[-1], activation='softmax', name='self_attention_probs_' + str(index))(input_tensor)
        attention_mul = Multiply(name='self_attention_mul_' + str(index))([input_tensor, attention_probs])
        return attention_mul

    def fit(self, X_train1, X_train2, y_train, x_val1, x_val2, y_val):
        input1 = Input(shape=(X_train1.shape[1],))
        input2 = Input(shape=(X_train2.shape[1],))

        processed1 = Reshape((-1, X_train1.shape[1]))(input1)
        processed1 = Conv1D(192, 3, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Reshape((-1, 192))(processed1)  # Flatten the tensor for self-attention
        processed1 = self.self_attention(processed1, 1)  # Apply self-attention mechanism
        processed1 = Conv1D(192, 3, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)

        processed2 = Reshape((-1, X_train2.shape[1]))(input2)
        processed2 = Conv1D(192, 1, strides=1, padding="same")(processed2)
        processed2 = BatchNormalization()(processed2)
        processed2 = LeakyReLU()(processed2)
        processed2 = Reshape((-1, 192))(processed2)  # Flatten the tensor for self-attention
        processed2 = self.self_attention(processed2, 2)  # Apply self-attention mechanism

        merged = Concatenate()([processed1, processed2])
        merged = Conv1D(192, 1, strides=1, padding="same")(merged)
        merged = BatchNormalization()(merged)
        merged = LeakyReLU()(merged)
        merged = MaxPool1D(1, strides=1)(merged)

        lstm_out = Bidirectional(LSTM(self.units, dropout=self.dropout,
                                      recurrent_dropout=0, kernel_regularizer=keras.regularizers.l2(0.01),
                                      return_sequences=False))(merged)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(self.dropout)(lstm_out)
        predictions = Dense(y_train.shape[1])(lstm_out)

        model = Model(inputs=[input1, input2], outputs=predictions)
        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        checkpoint = ModelCheckpoint(filepath="result/temp/best_model/best_model.keras", monitor='val_loss',
                                     verbose=1, save_best_only='True', mode='min')

        model.compile(loss='mse', optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])

        history = model.fit([X_train1, X_train2], y_train, epochs=self.num_epochs, verbose=self.verbose,
                            validation_data=([x_val1, x_val2], y_val), callbacks=[reduce_lr, checkpoint],
                            batch_size=self.batch_size)

        self.model = load_model("result/temp/best_model/best_model.keras")

        return history

    def predict(self, X_test1, X_test2):
        y_test_predicted = self.model.predict([X_test1, X_test2])
        return y_test_predicted




class CNN_lstm_v2_lfp_ff_attention(object):
    def __init__(self, units=150, dropout=0.5, num_epochs=50, verbose=1, batch_size=128, num_heads=8, model_path="result/temp/best_model/best_model2.keras"):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.model_path = model_path

    def fit(self, X_train1, X_train2, y_train, x_val1, x_val2, y_val):
        
        input1 = Input(shape=(X_train1.shape[1],))
        input2 = Input(shape=(X_train2.shape[1],))

        
        processed1 = Reshape((-1, X_train1.shape[1]))(input1)
        processed1 = Conv1D(192, 3, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(192, 3, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(192, 4, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(192, 4, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)

        # 第二个输入的处理流程
        processed2 = Reshape((-1, X_train2.shape[1]))(input2)
        processed2 = Conv1D(192, 3, strides=1, padding="same")(processed2)
        processed2 = BatchNormalization()(processed2)
        processed2 = LeakyReLU()(processed2)
        processed2 = Conv1D(192, 2, strides=1, padding="same")(processed2)
        processed2 = BatchNormalization()(processed2)
        processed2 = LeakyReLU()(processed2)

        # 合并两个输入的处理流程
        merged = Concatenate()([processed1, processed2])
        # merged = Conv1D(192, 1, strides=1, padding="same")(merged)
        # merged = BatchNormalization()(merged)
        # merged = LeakyReLU()(merged)

        # 自注意力机制
        attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=64)(merged, merged)
        merged = Concatenate()([merged, attention])

        merged = MaxPool1D(1, strides=1)(merged)

        # LSTM层
        lstm_out = Bidirectional(LSTM(self.units, dropout=self.dropout,
                                      recurrent_dropout=0, kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      return_sequences=False))(merged)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(self.dropout)(lstm_out)
        predictions = Dense(y_train.shape[1])(lstm_out)

        # 创建模型
        model = Model(inputs=[input1, input2], outputs=predictions)

        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        checkpoint = ModelCheckpoint(filepath=self.model_path, monitor='val_loss', verbose=1,
                                     save_best_only='True', mode='min')

        # 编译模型
        model.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        # 训练模型
        history = model.fit([X_train1, X_train2], y_train, epochs=self.num_epochs, verbose=self.verbose,
                            validation_data=([x_val1, x_val2], y_val), callbacks=[reduce_lr, checkpoint],
                            batch_size=self.batch_size)

        # 加载最佳模型
        self.model = load_model(self.model_path)

        return history

    def predict(self, X_test1, X_test2):
        # 预测
        y_test_predicted = self.model.predict([X_test1, X_test2])
        return y_test_predicted


class CNN_lstm_v2_lfp_ff_attention_BH(object):
    def __init__(self, units=100, dropout=0, num_epochs=100, verbose=1, batch_size=128, num_heads=8):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_heads = num_heads

    def fit(self, X_train1, X_train2, y_train, x_val1, x_val2, y_val):
        input1 = Input(shape=(X_train1.shape[1],))
        input2 = Input(shape=(X_train2.shape[1],))

        processed1 = Reshape((-1, X_train1.shape[1]))(input1)
        processed1 = Conv1D(80, 3, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(80, 3, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(80, 4, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)
        processed1 = Conv1D(80, 4, strides=1, padding="same")(processed1)
        processed1 = BatchNormalization()(processed1)
        processed1 = LeakyReLU()(processed1)

        processed2 = Reshape((-1, X_train2.shape[1]))(input2)
        processed2 = Conv1D(80, 1, strides=1, padding="same")(processed2)
        processed2 = BatchNormalization()(processed2)
        processed2 = LeakyReLU()(processed2)

        merged = Concatenate()([processed1, processed2])
        merged = Conv1D(80, 1, strides=1, padding="same")(merged)
        merged = BatchNormalization()(merged)
        merged = LeakyReLU()(merged)

        attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=64)(merged, merged)
        merged = Concatenate()([merged, attention])

        merged = MaxPool1D(1, strides=1)(merged)

        lstm_out = Bidirectional(LSTM(self.units, dropout=self.dropout,
                                      recurrent_dropout=0, kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      return_sequences=False))(merged)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(self.dropout)(lstm_out)
        predictions = Dense(3, activation='softmax')(lstm_out)  # 修改输出层为三分类

        model = Model(inputs=[input1, input2], outputs=predictions)
        model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        checkpoint = ModelCheckpoint(filepath="result/temp/best_model/best_model.keras", monitor='val_loss', verbose=1,
                                     save_best_only='True', mode='min')

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        history = model.fit([X_train1, X_train2], y_train, epochs=self.num_epochs, verbose=self.verbose,
                            validation_data=([x_val1, x_val2], y_val), callbacks=[reduce_lr, checkpoint],
                            batch_size=self.batch_size)

        self.model = load_model("result/temp/best_model/best_model.keras")

        return history

    def predict(self, X_test1, X_test2):
        y_test_predicted = self.model.predict([X_test1, X_test2])
        return y_test_predicted
