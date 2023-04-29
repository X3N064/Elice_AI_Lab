import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Sequential, Input

class EncoderDecoder(Model):
    def __init__(self, hidden_dim, encoder_input_shape, decoder_input_shape, num_classes):
        super(EncoderDecoder, self).__init__()
        
        # TODO: [지시사항 1번] SimpleRNN으로 이루어진 Encoder를 정의하세요.
        self.encoder = layers.SimpleRNN(hidden_dim, return_state=True, input_shape=encoder_input_shape)
                                        
        # TODO: [지시사항 2번] SimpleRNN으로 이루어진 Decoder를 정의하세요.
        self.decoder = layers.SimpleRNN(hidden_dim, return_sequences=True, input_shape=decoder_input_shape)
        
        self.dense = layers.Dense(num_classes, activation="softmax")
        
    def call(self, encoder_inputs, decoder_inputs):
        # TODO: [지시사항 3번] Encoder에 입력값을 넣어 Decoder의 초기 state로 사용할 state를 얻어내세요.
        _, encoder_state = self.encoder(encoder_inputs)
        
        # TODO: [지시사항 4번] Decoder에 입력값을 넣고, 초기 state는 Encoder에서 얻어낸 state로 설정하세요.
        decoder_outputs = self.decoder(decoder_inputs, initial_state=[encoder_state])
        
        outputs = self.dense(decoder_outputs)
        
        return outputs


def main():
    # hidden state의 크기
    hidden_dim = 20
    
    # Encoder에 들어갈 각 데이터의 모양
    encoder_input_shape = (10, 1)
    
    # Decoder에 들어갈 각 데이터의 모양
    decoder_input_shape = (30, 1)
    
    # 분류한 클래스 개수
    num_classes = 5

    # Encoder-Decoder 모델을 만듭니다.
    model = EncoderDecoder(hidden_dim, encoder_input_shape, decoder_input_shape, num_classes)
    
    # 모델에 넣어줄 가상의 데이터를 생성합니다.
    encoder_x, decoder_x = tf.random.uniform(shape=encoder_input_shape), tf.random.uniform(shape=decoder_input_shape)
    encoder_x, decoder_x = tf.expand_dims(encoder_x, axis=0), tf.expand_dims(decoder_x, axis=0)
    y = model(encoder_x, decoder_x)

    # 모델의 정보를 출력합니다.
    model.summary()

if __name__ == "__main__":
    main()