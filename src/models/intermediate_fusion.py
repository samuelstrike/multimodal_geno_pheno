from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, concatenate, BatchNormalization, Conv1D, Activation, MaxPooling1D, Flatten, Embedding, Normalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def intermediate_fusion(input_lstm, input_cnn, input_study):

    lstm_input = Input(shape = input_lstm, name = "LSTM_Input (Weather Data)")
    cnn_input = Input(shape = input_cnn, name = "CNN_Input (Gene Data)")
    study_input = Input(shape = input_study, name = "Study_Input (Treatment to Crop)")

    study_embedded = Embedding(input_dim=6, output_dim=3)(study_input)
    study_embedded = Flatten()(study_embedded)

    # Implement the intermediate fusion of the two inputs
    x_lstm = LSTM(223, return_sequences=True, name="LSTM_Layer_1")(lstm_input)
    x_lstm = Dropout(0.2)(x_lstm)
    x_lstm = LSTM(265, return_sequences = False, name="LSTM_Layer_2")(x_lstm)
    x_lstm = Dropout(0.2)(x_lstm)
    contact_study = concatenate([x_lstm,study_embedded], name="Concat_LSTM_Study")
    contact_study = BatchNormalization()(contact_study)
    x1 = Dense(136, activation='relu')(contact_study)
    x1 = Dropout(0.3)(x1)
    lstm_output = Dense(128, activation='relu', name="LSTM_Output")(x1)

    x = Conv1D(filters=64,kernel_size=5,padding='same',kernel_regularizer='L1L2', strides = 2, name="Conv1D_Layer_1")(cnn_input)
    x = Conv1D(filters=128,kernel_size=5,padding='same',kernel_regularizer='L1L2',strides = 1, name="Conv1D_Layer_2")(x)
    x = Conv1D(filters= 32,kernel_size=2,padding='same',kernel_regularizer='L1L2',strides = 1, name="Conv1D_Layer_3")(x)
    x = Conv1D(filters= 64,kernel_size=2,padding='same',kernel_regularizer='L1L2',strides = 1, name="Conv1D_Layer_4")(x)
    x= BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(0.87)(x)
    x= Activation('relu')(x)
    cnn_output = Dense(128)(x)

    concatenated = concatenate([lstm_output, cnn_output], name="Concat_LSTM_CNN")

    normalizer = Normalization()
    normalized_concat = normalizer(concatenated)

    z = Dense(128, activation='relu')(normalized_concat)
    z = Dropout(0.2)(z)
    final_output = Dense(1)(z)

    intermediate_fusion = Model(inputs=[lstm_input, cnn_input, study_input], outputs=final_output)

    optimizer = Adam(learning_rate=0.001)
    # optimizer = SGD(learning_rate=0.0009, momentum=0.77)

    intermediate_fusion.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return intermediate_fusion