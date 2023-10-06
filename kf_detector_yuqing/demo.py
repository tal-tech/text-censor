from kf_sentence_detector_yuqing import KFSentenceDetectorYuqing

with open('./words_yuqing.txt','r') as f:
    words = [line.strip('\n') for line in f.readlines()]
model_config_path = './config_yuqing.json'
kf_detector = KFSentenceDetectorYuqing(words, model_config_path)


if __name__ == '__main__':

    input_text = '如果今天开课前不解决。就投诉到消协,那我就等电话了，12点前不打电话，我打12315，你们课程和描述的不一样,打消协投诉的话是投诉我们要去的小白楼校区吗？'
    # input_text2 = '我要去法院投诉举报'

    result = kf_detector.predict_text(input_text)
    print(result)
