fs = 6000;    %采样频率
length = 0.5;    %时间长度(秒） 
% 创建一个录音文件：fs =7000Hz, 16-bit, 单通道
recorderObj = audiorecorder(fs, 16, 2);   
recordblocking(recorderObj, length); % 录音0.5秒钟
stop(recorderObj);
y = getaudiodata(recorderObj);
ymax = max(abs(y)); % 归一化
y = y/ymax;
audiowrite('..\system_design\send.wav', y, fs); % 存储录音文件
