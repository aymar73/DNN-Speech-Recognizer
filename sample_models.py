from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Building a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Adding recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Adding softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specifying the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Building a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Adding recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # Adding batch normalization 
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # Adding a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Adding softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specifying the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Building a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Adding convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Adding batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Adding a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, name='rnn')(bn_cnn) #implementation=2,
    # Adding batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # Adding a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Adding softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specifying the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Computing the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Building a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # recurrent layers, each with batch normalization
    previous_data = input_data
    simp_rnn = []
    bn_cnn = []
    for i in range(0, recur_layers):
        simp_rnn_active = GRU(units, return_sequences=True, implementation=2, name='rnn'+str(i))(previous_data)
        simp_rnn.append(simp_rnn_active)

        # Batch normalization
        bn_cnn_active = BatchNormalization(name="bn_conv_1d"+str(i))(simp_rnn[i])
        bn_cnn.append(bn_cnn_active)
        previous_input = bn_cnn[i]
        
    # Adding a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_cnn[recur_layers-1])
    # Adding softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specifying the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Building a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Adding bidirectional recurrent layer
    #bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation = 2, name = 'rnn'))(input_data)
    bidir_rnn = Bidirectional(SimpleRNN(units, activation='relu', return_sequences=True, name='rnn'))(input_data)
    # Adding a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Adding softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specifying the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29, dropout_rate=0.5,
                number_of_layers=2): #, activation='relu'
    """ Building a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Adding convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data) 
    
    # Adding batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    
    # Adding a recurrent layer
    bidir_rnn_1 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir_rnn_1'))(bn_cnn)
    
    # Adding batch normalization
    bn_rnn_1 = BatchNormalization(name='bn_rnn_1')(bidir_rnn_1)
    
    #bidir_rnn_2 = GRU(units, activation='relu',
    #    return_sequences=True, implementation=2, name='bidir_rnn_2', 
    #        recurrent_dropout=0.2, dropout=0.2)(bn_rnn_1)          #Bidirectional(, merge_mode='concat')
    # Add batch normalization
    #bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(bidir_rnn_2)
    
    #bidir_rnn_3 = GRU(units, activation='relu',
    #    return_sequences=True, implementation=2, name='bidir_rnn_3', 
    #        recurrent_dropout=0.2, dropout=0.2)(bn_rnn_2)          #Bidirectional(, merge_mode='concat')
    # Add batch normalization
    #bn_rnn_3 = BatchNormalization(name='bn_rnn_3')(bidir_rnn_3)
        
    time_dense1 = TimeDistributed(Dense(output_dim))(bn_rnn_1)
    
    # Dropout
    dropout = Dropout(0.4) (time_dense1)
    
    # Time distributed
    time_dense2 = TimeDistributed(Dense(output_dim))(dropout)
    
    # Adding softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense2)
    
    # Specifying the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    # Specifying model.output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    
    print(model.summary())
    
    return model
