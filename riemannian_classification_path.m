clear all
close all
tLen = 4; %78.41
delay = 2;



sub = 16;
%% Load data
[S_all, H_all] = loaddata(sub); %Returns cells of data from all available sessions
nbrSessions = length(S_all);
%% Preprocessing of all available sessions (Same for training and test data)
% 1) Band pass filter
for session = 1:nbrSessions
    x_all{session} = bandpass_filter_ext([12.95 13.05], [16.9 17.1], [20.9 21.1], S_all{session}, H_all{session}); %74.31
end

% Ploting best classified covariance matrices
%*************************************************************************

% 2) Rearange data per trial
X = get_trials(x_all, H_all, tLen, delay);

% 3) Covariance matrices of all trials summed up per class
Nt = size(X{1},3); %Number of trial
    for k = 1:Nt %loop for evrey trial
        for cl = 1:4
            if k == 1
                    P{cl}(:,:,k) = shcovft((X{cl}(:,:,k))'); % J. Schaefer Shrinkage covariance from Barachant toolbox
            else
                tmp = shcovft((X{cl}(:,:,k))'); % J. Schaefer Shrinkage covariance from Barachant toolbox
                P{cl} = cat(3, P{cl}(:,:,:), tmp); % 
            end
        end
    end
A = cat(3, P{1}, P{2}, P{3}, P{4});
B = mean_covariances(A,'riemann'); 
for cl = 1:size(P,2)
    Pc(:,:,cl) = mean_covariances(P{cl},'riemann');
end
Tc = Tangent_space(Pc,B); 
Tcc = Tangent_space(B,B); %The center of all matrices on the tangent space
T = Tangent_space(A,B);
[coef, score, latent] = pca(T');

tan_pca_x = reshape(score(:,1), numel(score(:,1))/4, 4);
tan_pca_y = reshape(score(:,2), numel(score(:,2))/4, 4); 
tan_pca_z = reshape(score(:,3), numel(score(:,3))/4, 4);

pca_c = Tc'*coef;
pca_c_x = reshape(pca_c(:,1), numel(pca_c(:,1))/4, 4);
pca_c_y = reshape(pca_c(:,2), numel(pca_c(:,2))/4, 4);
pca_c_z = reshape(pca_c(:,3), numel(pca_c(:,3))/4, 4);

pca_cc = Tcc'*coef;
pca_cc_x = pca_cc(:,1);
pca_cc_y = pca_cc(:,2);
pca_cc_z = pca_cc(:,3);

colors = ['b' 'g' 'r' 'c'];
figure
plot(tan_pca_x, tan_pca_y,'*'), legend(['Resting class'; '13Hz class   '; '21Hz class   '; '17Hz class   ']);
hold on
for cl = 1:size(P,2)
    plot(pca_c_x(cl), pca_c_y(cl),[colors(cl) 'o'], 'LineWidth', 4),% legend(['no-ssvep  '; '13Hz class'; '17Hz class'; '21Hz class']);
end
for c = 1:size(tan_pca_x,2)
    for tr = 1:size(tan_pca_x,1)
        plot([tan_pca_x(tr,c) pca_c_x(c)],[tan_pca_y(tr,c) pca_c_y(c)], colors(c));
    end
end
%plot(pca_cc_x, pca_cc_y,'kp', 'LineWidth', 4); %Plot riem mean of all matrices
%set(gca,'XTickLabel','','YTickLabel','');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
xlim = get(gca,'XLim');
%legend boxoff

%Repeat the ploting with class mean obtained from matrices filtered  from outlier
%********************************************************************************


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % The following lines try the geometric mean, std, and z-score as proposed
% % in Congedo HDR
% 
% % For each class,
%     for cl = 1:4
%         % get mean of class
%         P_filt{cl} = P{cl};
%         cont = 1;
%         while cont == 1
%             dis = [];
%             z = [];
%             Bc(:,:,cl) = mean_covariances(P_filt{cl},'riemann');
%             % get distance of each matrice to it class mean
%             for i = 1:size(P_filt{cl},3)
%                 dis(i,cl) = distance(P_filt{cl}(:,:,i), Bc(:,:,cl), 'riemann');
%             end
%             % get the geometric mean of the distances
%             mu(cl) = exp( mean(log(dis(:,cl))) );
%             % get geometric standard dev
%             sig(cl) = exp( sqrt(mean((log(dis(:,cl)/mu(cl))).^2)) );
%             % get the z-score
%             z(:,cl) = log(dis(:,cl)/mu(cl))/log(sig(cl));
%             % Threshold z-score
%             z_th(cl) = 1.5*sig(cl);
%             % Identify outliers (lying beyond z_th)
%             [outliers{cl} ind_out{cl}] = find(z(:,cl) > z_th(cl));
%             if isempty(ind_out{cl})
%                 %P_filt{cl} = P{cl};
%                 %P{cl} = P{cl};
%                 cont = 0;
%             else
%                 P_filt{cl} = P_filt{cl}(:,:, setxor([1:size(P_filt{cl},3)], outliers{cl})); 
%                 %P{cl} = P{cl}(:,:, setxor([1:size(P{cl},3)], outliers{cl}));
%                 Bc(:,:,cl) = mean_covariances(P_filt{cl},'riemann');
%             end 
%             %Bc_filt(:,:,cl) = mean_covariances(P_filt{cl},'riemann');       
%         end
%         Bc_filt(:,:,cl) = Bc(:,:,cl);
%         %P_filt{cl} = P{cl};
%     end
%         
%     A_filt = cat(3, P_filt{1}, P_filt{2}, P_filt{3}, P_filt{4});
%     B_filt = mean_covariances(A_filt,'riemann');
%     %T_filt1 = Tangent_space(A_filt, B);
%     T_filt1 = Tangent_space(A_filt, B_filt);
%     [coef_filt1, score_filt1, latent_filt1] = pca(T_filt1');
% 
%     nt1 = size(P_filt{1},3);
%     nt2 = size(P_filt{2},3);
%     nt3 = size(P_filt{3},3);
%     nt4 = size(P_filt{4},3);
% 
%     tan_pca_x_filt_class1 = score_filt1(1:nt1,1);
%     tan_pca_x_filt_class2 = score_filt1(nt1+1:nt1+nt2,1);
%     tan_pca_x_filt_class3 = score_filt1(nt1+nt2+1:nt1+nt2+nt3,1);
%     tan_pca_x_filt_class4 = score_filt1(nt1+nt2+nt3+1:nt1+nt2+nt3+nt4,1);
% 
%     tan_pca_y_filt_class1 = score_filt1(1:nt1,2);
%     tan_pca_y_filt_class2 = score_filt1(nt1+1:nt1+nt2,2);
%     tan_pca_y_filt_class3 = score_filt1(nt1+nt2+1:nt1+nt2+nt3,2);
%     tan_pca_y_filt_class4 = score_filt1(nt1+nt2+nt3+1:nt1+nt2+nt3+nt4,2);
% 
%     %Tc_filt = Tangent_space(Bc_filt,B);
%     Tc_filt = Tangent_space(Bc_filt,B_filt);
%     pca_c_filt = Tc_filt'*coef_filt1;
%     %pca_c_filt = Tc_filt'*coef_filt;
%     %pca_c_filt = score_filt_c;
%     pca_c_x_filt = reshape(pca_c_filt(:,1), numel(pca_c_filt(:,1))/4, 4);
%     pca_c_y_filt = reshape(pca_c_filt(:,2), numel(pca_c_filt(:,2))/4, 4);
%     pca_c_z_filt = reshape(pca_c_filt(:,3), numel(pca_c_filt(:,3))/4, 4);
% 
%     % pca_cc_filt = Tcc_filt'*coef_filt1;
%     % pca_cc_x_filt = pca_cc_filt(:,1);
%     % pca_cc_y_filt = pca_cc_filt(:,2);
%     % pca_cc_z_filt = pca_cc_filt(:,3);
% 
%     colors = ['b' 'g' 'r' 'c'];
%     figure
%     plot(tan_pca_x_filt_class1, tan_pca_y_filt_class1, 'b*')
%     hold on
%     plot(tan_pca_x_filt_class2, tan_pca_y_filt_class2, 'g*')
%     plot(tan_pca_x_filt_class3, tan_pca_y_filt_class3, 'r*')
%     plot(tan_pca_x_filt_class4, tan_pca_y_filt_class4, 'c*')
% 
%     hold on
%     for cl = 1:size(P_filt,2)
%         if cl==4
%             plot(pca_c_x_filt(cl), pca_c_y_filt(cl),[colors(cl) 'o'], 'LineWidth', 4),
%         else
%             plot(pca_c_x_filt(cl), pca_c_y_filt(cl),[colors(cl) 'o'], 'LineWidth', 4),% legend(['no-ssvep  '; '13Hz class'; '17Hz class'; '21Hz class']);
%         end
%     end
% 
%     for tr = 1:size(tan_pca_x_filt_class1,1)
%         plot([tan_pca_x_filt_class1(tr) pca_c_x_filt(1)],[tan_pca_y_filt_class1(tr) pca_c_y_filt(1)], colors(1));
%     end
%     for tr = 1:size(tan_pca_x_filt_class2,1)
%         plot([tan_pca_x_filt_class2(tr) pca_c_x_filt(2)],[tan_pca_y_filt_class2(tr) pca_c_y_filt(2)], colors(2));
%     end
%     for tr = 1:size(tan_pca_x_filt_class3,1)
%         plot([tan_pca_x_filt_class3(tr) pca_c_x_filt(3)],[tan_pca_y_filt_class3(tr) pca_c_y_filt(3)], colors(3));
%     end
%     for tr = 1:size(tan_pca_x_filt_class4,1)
%         plot([tan_pca_x_filt_class4(tr) pca_c_x_filt(4)],[tan_pca_y_filt_class4(tr) pca_c_y_filt(4)], colors(4));
%     end
% 
%     % plot(pca_cc_x_filt, pca_cc_y_filt,'kp', 'LineWidth', 4); %Plot riem mean of all matrices
%     %set(gca,'FontSize',16,'fontWeight','bold','outErrors{n}','','YTickLabel','')
%     set(gca,'FontSize',14,'fontWeight','normal')
%     set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','bold')
%     
% %  %########################################################################
% %  figure
% %     plot(tan_pca_x_filt_class1, -tan_pca_y_filt_class1, 'b*')
% %     hold on
% %     plot(tan_pca_x_filt_class2, -tan_pca_y_filt_class2, 'g*')
% %     plot(tan_pca_x_filt_class3, -tan_pca_y_filt_class3, 'r*')
% %     plot(tan_pca_x_filt_class4, -tan_pca_y_filt_class4, 'c*')
% % 
% %     hold on
% %     for cl = 1:size(P_filt,2)
% %         if cl==4
% %             plot(pca_c_x_filt(cl), -pca_c_y_filt(cl),[colors(cl) 'o'], 'LineWidth', 4),
% %         else
% %             plot(pca_c_x_filt(cl), -pca_c_y_filt(cl),[colors(cl) 'o'], 'LineWidth', 4),% legend(['no-ssvep  '; '13Hz class'; '17Hz class'; '21Hz class']);
% %         end
% %     end
% % 
% %     for tr = 1:size(tan_pca_x_filt_class1,1)
% %         plot([tan_pca_x_filt_class1(tr) pca_c_x_filt(1)],-[tan_pca_y_filt_class1(tr) pca_c_y_filt(1)], colors(1));
% %     end
% %     for tr = 1:size(tan_pca_x_filt_class2,1)
% %         plot([tan_pca_x_filt_class2(tr) pca_c_x_filt(2)],-[tan_pca_y_filt_class2(tr) pca_c_y_filt(2)], colors(2));
% %     end
% %     for tr = 1:size(tan_pca_x_filt_class3,1)
% %         plot([tan_pca_x_filt_class3(tr) pca_c_x_filt(3)],-[tan_pca_y_filt_class3(tr) pca_c_y_filt(3)], colors(3));
% %     end
% %     for tr = 1:size(tan_pca_x_filt_class4,1)
% %         plot([tan_pca_x_filt_class4(tr) pca_c_x_filt(4)],-[tan_pca_y_filt_class4(tr) pca_c_y_filt(4)], colors(4));
% %     end
% % 
% %     % plot(pca_cc_x_filt, pca_cc_y_filt,'kp', 'LineWidth', 4); %Plot riem mean of all matrices
% %     %set(gca,'FontSize',16,'fontWeight','bold','outErrors{n}','','YTickLabel','')
% %     set(gca,'FontSize',14,'fontWeight','normal')
% %     set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','bold')
% %  %########################################################################
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Unomment from here till last line for plot submitted to neurocomp

for i=1:size(A,3)
    dis(i) = distance(A(:,:,i),B,'riemann');
end
d_m = mean(dis);
d_s = std(dis);
%d_th = d_m+2.5*d_s; %as defined by Barachant
d_th = d_m+1.117*d_s; %modifid to fit data
[outliers ind_out] = find(dis > d_th);
A_filt = A(:,:, setxor([1:size(A,3)], ind_out));
B_filt = mean_covariances(A_filt,'riemann');

%----------Get filtered P per class
%outliers_c = zeros(size(P,2), size(P{1},3));
%ind_out_c = zeros(size(P,2), size(P{1},3));
for i = 1:size(P,2)
    for j=1:size(P{1},3)
        dis_c(i,j) = distance(P{i}(:,:,j),B,'riemann');
    end
    [outliers_c{i} ind_out_c{i}] = find(dis_c(i,:) > d_th);
    P_filt{i} = P{i}(:,:, setxor([1:size(P{i},3)], ind_out_c{i}));
end
for cl = 1:size(P_filt,2)
    Pc_filt(:,:,cl) = mean_covariances(P_filt{cl},'riemann');
end

Tc_filt = Tangent_space(Pc_filt,B_filt); 
Tcc_filt = Tangent_space(B_filt,B_filt); %The center of all matrices on the tangent space
%T_filt = Tangent_space(A,B_filt); %I am using A instead of A_filt just to get all the 64 vectors on the tangent space. I will remove the outliers in the plot
T_filt1 = Tangent_space(A_filt, B_filt); %Let see this

%[coef_filt, score_filt, latent_filt] = pca(T_filt');
[coef_filt1, score_filt1, latent_filt1] = pca(T_filt1');

% tan_pca_x_filt = reshape(score_filt(:,1), numel(score_filt(:,1))/4, 4);
% tan_pca_y_filt = reshape(score_filt(:,2), numel(score_filt(:,2))/4, 4); 
% tan_pca_z_filt = reshape(score_filt(:,3), numel(score_filt(:,3))/4, 4);


nt1 = size(P_filt{1},3);
nt2 = size(P_filt{2},3);
nt3 = size(P_filt{3},3);
nt4 = size(P_filt{4},3);

tan_pca_x_filt_class1 = score_filt1(1:nt1,1);
tan_pca_x_filt_class2 = score_filt1(nt1+1:nt1+nt2,1);
tan_pca_x_filt_class3 = score_filt1(nt1+nt2+1:nt1+nt2+nt3,1);
tan_pca_x_filt_class4 = score_filt1(nt1+nt2+nt3+1:nt1+nt2+nt3+nt4,1);

tan_pca_y_filt_class1 = score_filt1(1:nt1,2);
tan_pca_y_filt_class2 = score_filt1(nt1+1:nt1+nt2,2);
tan_pca_y_filt_class3 = score_filt1(nt1+nt2+1:nt1+nt2+nt3,2);
tan_pca_y_filt_class4 = score_filt1(nt1+nt2+nt3+1:nt1+nt2+nt3+nt4,2);


% tan_pca_x_filt_class1 = score_filt1(1:size(P_filt{1},3),1);
% tan_pca_x_filt_class2 = score_filt1(size(P_filt{1},3)+1:size(P_filt{1},3)+size(P_filt{2},3),1);
% tan_pca_x_filt_class3 = score_filt1(size(P_filt{2},3)+1:size(P_filt{2},3)+size(P_filt{3},3),1);
% tan_pca_x_filt_class4 = score_filt1(size(P_filt{3},3)+1:size(P_filt{3},3)+size(P_filt{4},3),1);
% 
% tan_pca_y_filt_class1 = score_filt1(1:size(P_filt{1},3),2);
% tan_pca_y_filt_class2 = score_filt1(size(P_filt{1},3)+1:size(P_filt{1},3)+size(P_filt{2},3),2);
% tan_pca_y_filt_class3 = score_filt1(size(P_filt{2},3)+1:size(P_filt{2},3)+size(P_filt{3},3),2);
% tan_pca_y_filt_class4 = score_filt1(size(P_filt{3},3)+1:size(P_filt{3},3)+size(P_filt{4},3),2);


%---------------Remome outliers from plot replacing them by NaA
% for i = 1:4
%     tan_pca_x_filt(ind_out_c{i},i) = NaN;
%     tan_pca_y_filt(ind_out_c{i},i) = NaN;
%     tan_pca_z_filt(ind_out_c{i},i) = NaN;
% end

%[coef_filt_c, score_filt_c, latent_filt_c] = pca(Tc_filt');
pca_c_filt = Tc_filt'*coef_filt1;
%pca_c_filt = Tc_filt'*coef_filt;
%pca_c_filt = score_filt_c;
pca_c_x_filt = reshape(pca_c_filt(:,1), numel(pca_c_filt(:,1))/4, 4);
pca_c_y_filt = reshape(pca_c_filt(:,2), numel(pca_c_filt(:,2))/4, 4);
pca_c_z_filt = reshape(pca_c_filt(:,3), numel(pca_c_filt(:,3))/4, 4);

%pca_cc_filt = Tcc_filt'*coef_filt;
pca_cc_filt = Tcc_filt'*coef_filt1;
pca_cc_x_filt = pca_cc_filt(:,1);
pca_cc_y_filt = pca_cc_filt(:,2);
pca_cc_z_filt = pca_cc_filt(:,3);

colors = ['b' 'g' 'r' 'c'];
% figure
% plot(tan_pca_x_filt, tan_pca_y_filt,'*'), legend(['no-ssvep  '; '13Hz class'; '21Hz class'; '17Hz class']);

figure
plot(tan_pca_x_filt_class1, tan_pca_y_filt_class1, 'b*')
hold on
plot(tan_pca_x_filt_class2, tan_pca_y_filt_class2, 'g*')
plot(tan_pca_x_filt_class3, tan_pca_y_filt_class3, 'r*')
plot(tan_pca_x_filt_class4, -tan_pca_y_filt_class4, 'c*')

% tan_pca_x_filt_class1 = score_filt1(1:size(P_filt{1},3),1);
% tan_pca_x_filt_class2 = score_filt1(size(P_filt{1},3)+1:size(P_filt{1},3)+size(P_filt{2},3),1);
% tan_pca_x_filt_class3 = score_filt1(size(P_filt{2},3)+1:size(P_filt{2},3)+size(P_filt{3},3),1);
% tan_pca_x_filt_class4 = score_filt1(size(P_filt{3},3)+1:size(P_filt{3},3)+size(P_filt{4},3),1);
% 
% tan_pca_y_filt_class1 = score_filt1(1:size(P_filt{1},3),2);
% tan_pca_y_filt_class2 = score_filt1(size(P_filt{1},3)+1:size(P_filt{1},3)+size(P_filt{2},3),2);
% tan_pca_y_filt_class3 = score_filt1(size(P_filt{2},3)+1:size(P_filt{2},3)+size(P_filt{3},3),2);
% tan_pca_y_filt_class4 = score_filt1(size(P_filt{3},3)+1:size(P_filt{3},3)+size(P_filt{4},3),2);

hold on
for cl = 1:size(P_filt,2)
    if cl==4
        plot(pca_c_x_filt(cl), -pca_c_y_filt(cl),[colors(cl) 'o'], 'LineWidth', 4),
    else
        plot(pca_c_x_filt(cl), pca_c_y_filt(cl),[colors(cl) 'o'], 'LineWidth', 4),% legend(['no-ssvep  '; '13Hz class'; '17Hz class'; '21Hz class']);
    end
end

for tr = 1:size(tan_pca_x_filt_class1,1)
    plot([tan_pca_x_filt_class1(tr) pca_c_x_filt(1)],[tan_pca_y_filt_class1(tr) pca_c_y_filt(1)], colors(1));
end
for tr = 1:size(tan_pca_x_filt_class2,1)
    plot([tan_pca_x_filt_class2(tr) pca_c_x_filt(2)],[tan_pca_y_filt_class2(tr) pca_c_y_filt(2)], colors(2));
end
for tr = 1:size(tan_pca_x_filt_class3,1)
    plot([tan_pca_x_filt_class3(tr) pca_c_x_filt(3)],[tan_pca_y_filt_class3(tr) pca_c_y_filt(3)], colors(3));
end
for tr = 1:size(tan_pca_x_filt_class4,1)
    plot([tan_pca_x_filt_class4(tr) pca_c_x_filt(4)],-[tan_pca_y_filt_class4(tr) pca_c_y_filt(4)], colors(4));
end
% for c = 1:size(tan_pca_x_filt,2)
%     for tr = 1:size(tan_pca_x_filt,1)
%         plot([tan_pca_x_filt(tr,c) pca_c_x_filt(c)],[tan_pca_y_filt(tr,c) pca_c_y_filt(c)], colors(c));
%     end
% end


plot(pca_cc_x_filt, pca_cc_y_filt,'kp', 'LineWidth', 4); %Plot riem mean of all matrices
%----------This is done to center the new graph on a plot of the same scale
%as the previous
%xlim2 = get(gca,'XLim');
%extra = (xlim(2)-xlim(1)) - (xlim2(2)-xlim2(1));
%xlim3 = [xlim2(1)-extra/2, xlim2(2)+extra/2]
%set(gca,'XTickLabel','','YTickLabel','', 'XLim', xlim3);
%--------------------------------------------------------
set(gca,'FontSize',16,'fontWeight','bold','XTickLabel','','YTickLabel','')
set(findall(gcf,'type','text'),'FontSize',12,'fontWeight','bold')


% %***********************************************************************************
% figure(3)
% plot3(tan_pca_x, tan_pca_y, tan_pca_z,'*'), legend(['no-ssvep  '; '13Hz class'; '21Hz class'; '17Hz class']); 
% grid on;
% hold on
% for cl = 1:size(P,2)
%     plot3(pca_c_x(cl), pca_c_y(cl), pca_c_z(cl),[colors(cl) 's'], 'LineWidth', 4),% legend(['no-ssvep  '; '13Hz class'; '17Hz class'; '21Hz class']);
% end
% 
% %Classification Path
% %******************************************************
% tLen2 = 3.6; 
% delay = 0:0.2:2.4;
% for del = 1:numel(delay)
%     Xpath = get_trials(x_all, H_all, tLen2, delay(del));
%     % Covariance matrices of all trialssummed up per class
%     Nt = size(Xpath{1},3); %Number of trial
%     for k = 1:Nt %loop for evrey trial
%         for cl = 1:4
%             Ppath{cl}(:,:,k) = shcovft((Xpath{cl}(:,:,k))'); % J. Schaefer Shrinkage covariance from Barachant toolbox
%         end 
%     end
%     
%     COVtrain = A;
%     COVtest = cat(3, Ppath{1}, Ppath{2}, Ppath{3}, Ppath{4});
%     labels = [zeros(1,size(Ppath{1},3)) ones(1, size(Ppath{2},3)) 2*ones(1, size(Ppath{3},3)) 3*ones(1, size(Ppath{4},3))];
%     Ytrain = [zeros(1,size(P{1},3)) ones(1, size(P{3},3)) 2*ones(1, size(P{3},3)) 3*ones(1, size(P{4},3))];
%     % Classification by Remannian Distance
%     [Ytest(del,:) d C] = mdm(COVtest,COVtrain,Ytrain);    
%     %ac = sum((labels-Ytest)==0)/numel(labels);
%     
%     
%     Tpath = Tangent_space(COVtest,B); 
%     pca_path = Tpath'*coef;
%     pca_path_x(:,:,del) = reshape(pca_path(:,1), numel(pca_path(:,1))/4, 4);
%     pca_path_y(:,:,del) = reshape(pca_path(:,2), numel(pca_path(:,2))/4, 4);
%     pca_path_z(:,:,del) = reshape(pca_path(:,3), numel(pca_path(:,3))/4, 4);
% end
% 
% figure(3)
% hold on
% for k = 1:size(pca_path_x,1)
%     for i = 1:numel(delay)
%         plot(pca_path_x(k,2,i), pca_path_y(k,2,i),[colors(Ytest(i,40+k)+1) 'o'], 'LineWidth', 1)
%     end
% end
% 
% x = squeeze(pca_path_x(1,2,:));
% y = squeeze(pca_path_y(1,2,:));
% figure(4)
% plot(x, y,'*');
% hold on
% quiver(x(1:end-1),  y(1:end-1), ones(numel(x)-1,1), y(2:end) - y(1:end-1))