clear;
clc;

%%%%%%%%%%%%%%%%%%%%导入发送语音信号%%%%%%%%%%%%%%%%%%%%%
[source,Fs] = audioread('..\system_design\send.wav');
source = source';                  %导入语音信号为双声道信号
source_t = 1:1:length(source);     %原语音信号时间向量
figure(1)
plot(source_t,source);
xlabel('时间t')
ylabel('幅值')
title('读入语音信号的时域图')

%%%%%%%%%%%%%%%%%%%%%%预测器差分环节%%%%%%%%%%%%%%%%%%%%%
error = zeros(1,length(source));              %当前值-预测值的差分值
predict = 0;                                  %预测值初始为0
for i = 1:length(source)
    error(1,i) = source(1,i) - predict;  %当前样值-预测值
    predict = predict + error(1,i);           %更新预测其为上一次样值
end

%%%%%%%%%%%%%%%%%%%信号输入归一化环节%%%%%%%%%%%%%%%%%%%
sign_of_signal=sign(error);         %存储差分值的极性信息

%A律13折线的电平范围为0~2048
%因此需要先将抽样值归一化
%再全部乘以2048，将信号幅值控制在A律量化电平范围内
MaxS=max(abs(error));               
Norm_signal=abs(error/MaxS);        
input_signal=2048*Norm_signal;          
    
%%%%%%%%%%%%%非均匀量化编码环节（A律13折线）%%%%%%%%%%%%%%%
%存储并行的8位PCM码值：C1极性码 C2C3C4段落码 C5C6C7 段内码
code_value = zeros(length(input_signal),8);     

% 段落码编码程序
for i=1:length(input_signal)
    if (input_signal(i)>=128)&&(input_signal(i)<=2048)
        code_value(i,2)=1;                  %在第五段与第八段之间，段位码第一位都为"1"
    end
    if (input_signal(i)>32)&&(input_signal(i)<128)||...
        (input_signal(i)>=512)&&(input_signal(i)<=2048)
        code_value(i,3)=1;                  %在第三四七八段内，段位码第二位为"1"
    end
    if (input_signal(i)>=16)&&(input_signal(i)<32)||...
        (input_signal(i)>=64)&&(input_signal(i)<128)||...
        (input_signal(i)>=256)&&(input_signal(i)<512)||...
        (input_signal(i)>=1024)&&(input_signal(i)<=2048)
            code_value(i,4)=1;              %在二四六八段内，段位码第三位为"1"
    end
end

%计算每个信号值量化后的段落序号
N=zeros(1,length(input_signal));                   %存储每个量化值的段落序号             
for i=1:length(input_signal)
    duanluo_code = num2str(code_value(i,2:4));     %将3位段落码打包到一个字符数组中
    N(1,i) = bin2dec(duanluo_code) + 1;            %将二进制段落码转化为十进制数字 找到对应的段落序号
end

%A律13折线使用8位编码 256个量化级 分为8个段落 每个段落内又划为16个量化级
start_level = [0,16,32,64,128,256,512,1024];             %每一个段落的起始电平
delta_V = [1/16,1/16,2/16,4/16,8/16,16/16,32/16,64/16];  %每个段落的最小量化间隔

%段内码编码程序
for i=1:length(input_signal)
    res = input_signal(i)-start_level(N(1,i)); %用段落序号索引起始电平 计算段内十进制数值
    q = ceil(res/(delta_V(N(i))*16));          %用段落序号索引量化间隔 计算段内量化级    
    if q==0
        code_value(i,(5:8))=[0,0,0,0];         %刚好落在起始电平 段内码编码全0
    else 
        %k是存储段内码二进制的字符数组
        k=num2str(dec2bin(q-1,4));             %将段内码十进制数值根据权重编为二进制
        code_value(i,5)=str2double(k(1));    
        code_value(i,6)=str2double(k(2));
        code_value(i,7)=str2double(k(3));
        code_value(i,8)=str2double(k(4));
    end
    %极性码判断
    if sign_of_signal(i)>0
        code_value(i,1)=1;
    elseif sign_of_signal(i)<0
        code_value(i,1)=0;
    end                                          
end

%将并行的8位PCM码组转换为串行的码流送入调制器
PCM_code_send = reshape(code_value', 1, []);
audiowrite('..\system_design\pcm_code.wav',PCM_code_send,Fs);

%%%%%%%%%%%%%%差分编码（绝对码变为相对码）%%%%%%%%%%%%%%%
diff_code_stream = PCM_code_send;
code_num = length(PCM_code_send);
for i = 2:code_num
    if abs(diff_code_stream(i) - diff_code_stream(i-1)) == 1
        diff_code_stream(i) = 1;
    else
        diff_code_stream(i) = 0;
    end
end

%%%%%%%%%%%%%%%%%%%%%%2DPSK键控调制%%%%%%%%%%%%%%%%%%%    
code_Ts = 1;                          %码元宽度
Rb = 1/code_Ts;                       %码元速率
L = 100;
dt = 1/L;                             %采样间隔
total_t = code_num * code_Ts;         %码流传输总时间
code_t = 0:dt:total_t-dt;             %码元时间向量
fc = 2 * Rb;           %高频载波频率是码元速率的2倍     
phase_1 = 0;           %存储码元为"1"时的已调信号相位   
phase_2 = 0;           %存储码元为"0"时的已调信号相位
BDPSK_signal = [];      %存储二进制相位调制的已调信号
% NRZ = [];              %存储双极性不归零矩形波形
% diff_NRZ = [];         %存储双极性不归零的差分编码波形
phase = [];            %存储相位调制时相位变化信息
carrier_wave = [];     %存储高频载波

for i = 1:code_num
  %一个码元对应的时间尺度 
  t1 = (i-1)*code_Ts:dt:i*code_Ts-dt;
  %存储相位变化标志 "1"码相位为0度
  if (diff_code_stream(i)==1)
      phase_1 = ones(1,L);        
      phase = [phase,phase_1];
  end
  %存储相位变化标志 "0"码相位为180度
  if (diff_code_stream(i)==0)
      phase_0 = -ones(1,L);         
      phase = [phase,phase_0];
  end
  c = sin(2*pi*fc*t1);              %高频载波
  carrier_wave = [carrier_wave,c];
end

BDPSK_signal = carrier_wave.*phase;   %让已调信号具有相位信息变化

%%%%%%%%%%%%%%%%%%%信道传输环节%%%%%%%%%%%%%%%%%
send_single = BDPSK_signal;              %发送已调信号
SNR = 10;                                %信噪比为20dB
trans_single= awgn(send_single,SNR);     %信道传输引入加性高斯噪声
receive_single = trans_single;           %接收信号

% %%%%%%%%%%%%%%%%%%%%解调环节%%%%%%%%%%%%%%%%%%%
decode_stream = zeros(1,L*code_num);      %存储五个恢复的数字信号码流

%低通滤波器参数设置  
wp=2*pi*3;ws=2*pi*4;Ap=3;As=30;  
[N,wc]=buttord(wp,ws,Ap,As,'s');
[num,den]=butter(N,wc,'s');

%相干解调
Demodem_signal = receive_single.*carrier_wave*2;
%低通滤波
LFP_Demodem_signal = lsim(tf(num,den),Demodem_signal,code_t);
for m = L/2:L:L*code_num
    %抽样判决
    if LFP_Demodem_signal(m) < 0
        for i = 1:L
            decode_stream((m-L/2)+i) = -1;
        end
        %抽样判决
    elseif LFP_Demodem_signal(m) >= 0
        for i = 1:L
            decode_stream((m-L/2)+i) = 1;
        end
    end
end

%%%%%%%%%%%%%%码反变换（相对码变为绝对码）%%%%%%%%%%%%%%%
true_code_stream = zeros(1,L*code_num);
true_code_stream(:,1:100) = decode_stream(:,1:100);
flag = 1;
for p = L+1:L:L*(code_num-1)
    if decode_stream(p) == decode_stream(p-L)
       for q = 1:L
           true_code_stream(p+q) = -1;
       end  
    else
        for q = 1:L
            true_code_stream(p+q) = 1;
        end
    end
end

final_stream = zeros(1,code_num); 
for q = L/2:L:L*code_num-L/2
    if true_code_stream(q) == -1
        final_stream((q+L/2)/L) = 0;
    elseif true_code_stream(q) == 1
        final_stream((q+L/2)/L) = 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%解码环节%%%%%%%%%%%%%%%%%%%%%%%%
%接收端接收PCM码流并做串行转换成并行数据处理
PCM_code_receive = (reshape(final_stream',8,length(final_stream)/8))';
[rows,cols] = size(PCM_code_receive);
decode_value = zeros(1,rows);              %译码处理后的信号幅度数值 
sign_of_rebuild_signal = zeros(1,rows);    %译码过程中的恢复的极性信息
for i=1:rows
    %根据第一位极性码恢复信号的极性信息
    if PCM_code_receive(i,1) == 1
        sign_of_rebuild_signal(1,i) = 1;
    elseif  PCM_code_receive(i,1) == 0
        sign_of_rebuild_signal(1,i) = -1;
    end
    %译码时根据段落码得到的段落序号
    M=bin2dec(num2str(PCM_code_receive(i,(2:4))))+1;
    %译码时将段内码二进制转换为十进制数
    Y=bin2dec(num2str(PCM_code_receive(i,(5:8))));
    if Y==0
        decode_value(1,i)=start_level(M);
    else
       decode_value(1,i)=start_level(M)+delta_V(M)*16*Y;
    end
end

%译码后的重建差分信号
rebuild_error = sign_of_rebuild_signal.*decode_value;
%重建c差分信号归一化后并乘以原信号的幅度信息，恢复原始的幅度信息
rebuild_error = rebuild_error/2048*MaxS;

%%%%%%%%%%%%%%%%%累加差分值恢复原信号环节%%%%%%%%%%%%%%%%%%
rebuild_signal = zeros(1,length(rebuild_error));  %重建信号
cnt = 0;                                          %接收端累加器初始为0
for i = 1:length(rebuild_error)
    rebuild_signal(1,i) = rebuild_error(1,i) + cnt; %当前样值-预测值
    cnt = rebuild_signal(1,i);                      %更新预测其为上一次样值
end

figure
plot(source_t,rebuild_signal);

audiowrite('..\system_design\receive.wav',rebuild_signal,2000);