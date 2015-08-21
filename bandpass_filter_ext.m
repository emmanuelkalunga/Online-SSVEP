function x = bandpass_filter_ext(f1, f2, f3, S, H)
Fs = H.SampleRate;
Fn = Fs/2;
if nargin == 3
    n_butter = 8;
    [b, a] = butter(n_butter, f1./Fn, 'bandpass');
    x = filtfilt(b, a, S); %Band pass
elseif nargin == 5
    Rp = 3; Rs = 10;
    Wp1 = f1/Fn; Ws1 = [f1(1)-1 f1(2)+1]/Fn;
    Wp2 = f2/Fn; Ws2 = [f2(1)-1 f2(2)+1]/Fn;
    Wp3 = f3/Fn; Ws3 = [f3(1)-1 f3(2)+1]/Fn;
    
%     Wp1 = f1/Fn; Ws1 = [f1(1)-0.1 f1(2)+0.1]/Fn;
%     Wp2 = f2/Fn; Ws2 = [f2(1)-0.1 f2(2)+0.1]/Fn;
%     Wp3 = f3/Fn; Ws3 = [f3(1)-0.1 f3(2)+0.1]/Fn;
    
    [n1,Wn1] = buttord(Wp1,Ws1,Rp,Rs);
    [n2,Wn2] = buttord(Wp2,Ws2,Rp,Rs);
    [n3,Wn3] = buttord(Wp3,Ws3,Rp,Rs);
    
    [b1, a1] = butter(n1, Wn1, 'bandpass');
    [b2, a2] = butter(n2, Wn2, 'bandpass');
    [b3, a3] = butter(n3, Wn3, 'bandpass');
    
    x1 = filtfilt(b1, a1, S); %Band pass
    x2 = filtfilt(b2, a2, S); %Band pass
    x3 = filtfilt(b3, a3, S); %Band pass
    x = [x1 x2 x3];
%     x = [x1+x2+x3 x1 x2 x3];
end

% [z,p,k] = butter(n1,Wn1);
% sos = zp2sos(z,p,k);
% freqz(sos,128,Fs)
% title(sprintf('n = %d Butterworth Bandpass Filter',n1))
% plot(x1(1:256,:));