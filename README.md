# WAV-to-TAB-Project

**WAV음원 파일 또는 YouTube 음원 링크로부터 기타 TAB 악보를 생성해내는 웹앱 프로젝트**

시연 영상(클릭)<br>
[![Video Label](http://img.youtube.com/vi/I-hnv9Tibk4/0.jpg)](https://youtu.be/I-hnv9Tibk4)

**프로젝트 과정**

**1.**
Guitarset (https://guitarset.weebly.com/)<br>
위 dataset의 WAV, JAMS -> MIDI 전처리 후 WAV-to-MIDI 변환 모델 학습.

**2.**
학습한 모델을 이용하여 WAV-to-MIDI 변환 코드 생성.

**3.**
tuttut (https://github.com/natecdr/tuttut?tab=MIT-1-ov-file)<br>
tuttut에서 제시하는 MIDI-to-TAB 알고리즘을 하나의 코드로 리팩터링.

**4.**
위의 과정들을 하나로 엮어 flask api구성.<br>

**5.**
streamlit으로 web app을 만들고 WAV음원 또는 YouTube링크로부터 음원을 추출하여 TAB 변환을 시도.


**※정확도를 보장하지 않습니다**
