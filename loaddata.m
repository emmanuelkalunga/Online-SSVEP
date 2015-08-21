% [S, H] = loaddata(subject, session)
% Description
% -----------
% Load all sessions data from gdf files using biosig function 'sload'
% Inputs
% - subject: Subject number (Id)
% - session: Number of available sessions
% Outputs
% -------
% - S: Cell containing multichanel signals from each seesion of subject 
% - H: Cell containing header files [as returned] from each seesion of subject
%       by sload function
% Auther: Emmanuel K. Kalunga
function [S, H] = loaddata(subject, session)
channels = 0;
disp('Loading...->>')
direct = ['./data/subject',num2str(subject),'/Training'];
fnames = dir(fullfile(direct));
nbrSessions = length(fnames) - 2;
if nargin == 1 %Load all session
    for session = 1:nbrSessions
        Filename = ['./data/subject',num2str(subject),'/Training/',fnames(session+2).name];
        [S{session}, H{session}] = sload(Filename, channels, 'OVERFLOWDETECTION:OFF');
    end
else
    Filename = ['./data/subject',num2str(subject),'/Training/',fnames(session+2).name];
    [S, H] = sload(Filename, channels, 'OVERFLOWDETECTION:OFF');
end