def final_model(input_dim, 
                 filters,
                 kernel_size,
                 conv_stride,
                 conv_border_mode,
                 units,
                 output_dim,
                 #drpout_W ,
                 #drpout_U 
               ):

    #same parameters as of cnn_rnn_model add dropout_W and dropout_U for RNN
    
     
     
    #Build a deep network for speech 
     
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    
    #ConvID Layers 
    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_1')(input_data)
    # Add batch normalization
    bn_cnn_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 ,name='bn_conv_1d_1')(conv_1d_1)
    #add Drop out to cnnn layer
    #dropout_1 = Dropout(0.2)(bn_cnn_1)
    
       
    #now build Recurrent layers
    # will be adding 4 layers
    
    #First Bidirectional RNN layer , BatchNormalization , DropOut
    deep_rnn_1 = Bidirectional(LSTM(units, activation='relu', return_sequences=True, implementation=2,
                                    #recurrent_dropout = drpout_W, dropout = drpout_U,
                                    name = 'deep_rnn_1'), merge_mode='concat')(bn_cnn_1)     

    
    
    BN_deep_rnn_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                                       name = 'BN_deep_rnn_1')(deep_rnn_1)
    
    #Dropout_deep_rnn_1 = Dropout(0.2)(BN_deep_rnn_1)
    
    
    #2nd Bidirectional RNN layer , BatchNormalization , DropOut
    deep_rnn_2 = Bidirectional(LSTM(units, activation='relu',
                                    return_sequences=True, implementation=2,
                                    #recurrent_dropout = drpout_W, dropout = drpout_U,
                                    name = 'deep_rnn_2') , 
                               merge_mode='concat')(BN_deep_rnn_1)     

    BN_deep_rnn_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                                       name = 'BN_deep_rnn_2')(deep_rnn_2)
    
    #Dropout_deep_rnn_2 = Dropout(0.2)(BN_deep_rnn_2)    
    
    #3rd Bidirectional RNN layer , BatchNormalization , DropOut
    deep_rnn_3 = Bidirectional(LSTM(units, activation='relu',
                                    return_sequences=True, implementation=2,
                                    #recurrent_dropout = drpout_W, dropout = drpout_U,
                                    name = 'deep_rnn_3') , 
                               merge_mode='concat')(BN_deep_rnn_2)     
    
    BN_deep_rnn_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                                       name = 'BN_deep_rnn_3')(deep_rnn_3)
   
    #Dropout_deep_rnn_3 = Dropout(0.2)(BN_deep_rnn_3)
    
    
    #4th Bidirectional RNN layer , BatchNormalization , DropOut
    #deep_rnn_4 = Bidirectional(LSTM(units, activation='relu',
    #                                return_sequences=True, implementation=2,
    #                                #recurrent_dropout = drpout_W, dropout = drpout_U,
    #                                name = 'deep_rnn_4') , 
    #                           merge_mode='concat')(BN_deep_rnn_3)     
    #
    #BN_deep_rnn_4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
    #                                   name = 'BN_deep_rnn_4')(deep_rnn_4)
    #Dropout_deep_rnn_4 = Dropout(0.2)(BN_deep_rnn_4)  
    
    
    #  TimeDistributed(Dense(output_dim)) 
    time_dense = TimeDistributed(Dense(output_dim))(BN_deep_rnn_3)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    # TODO: Specify model.output_length  
    #same as that of cnn_rnn_model 
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    
    
    print(model.summary())
    return model


