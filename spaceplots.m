function spaceplots(varargin)
%SPACEPLOTS: Reposition plots in a figure with custom spacing between them
% 
%   Usage: Draw your figure, with all the subplots, then call SPACEPLOTS. 
% 
%   SPACEPLOTS(fid,fpad,axpad) repositions plots in the figure given by
%   handle fid by adding the figure padding and axes padding given by fpad
%   and axpad respectively.
% 
%   "fpad" is a 4 element vector specifying (in normalized units)
%   the space to be left around the entire subplot grid. The format for
%   figure padding is [left right top bottom].
% 
%   "axpad" is a 2 element vector specifying (in normalized units) the
%   horizontal and vertical padding between the panels of the subplot grid.
%   The format for axes padding is [horizontal vertical].
% 
%   SPACEPLOTS(fpad,axpad) uses the current figure.
% 
%   SPACEPLOTS without any input arguments uses the current figure, and
%   assumes zero figure padding and axes padding.
% 
%   SPACEPLOTS simply repositions axes in a figure, it does not change any 
%   other properties of the figure. Also, in case of figures with multiple 
%   axes, SPACEPLOTS works only if these axes were created using the 
%   standard Matlab function "subplot".
% 
%NOTE: Please replace the default Matlab subplot.m file with the modified 
%version available with SPACEPLOTS v3.0 for the function to work properly.
%  
%version 3.0
%(c) Aditya Joshi, 2013


%% Parse Inputs

cargin = {};
if nargin > 0
    if ishandle(varargin{1})
        figure(varargin{1})
        cargin = varargin(2:end);
    else
        cargin = varargin;
    end
end

fpad = [0 0 0 0]; axpad = [0 0];
%fpad = [0.05 0.05 0 0.05]; axpad = [0.08 0.08]; %-- for plotyy
if numel(cargin) > 0, fpad = cargin{1}; end
if numel(cargin) > 1, axpad = cargin{2}; end
        
u1 = get(gcf,'units');      %to restore later
set(gcf,'units','normalized')

if isequal(getappdata(gcf,'spaceplots'),1)
    hAxGrid = getappdata(gcf,'SubplotGrid');
    if isempty(hAxGrid)
        error(['Cannot use SPACEPLOTS on this figure. An essential figure' ...
               ' property has been deleted, possibly because of a user operation.'])
    end
elseif isequal(numel(findobj(gcf,'type','axes','-and','-not','tag','legend')),1)
    hAxGrid = gca;
elseif isequal(numel(findobj(gcf,'type','axes','-and','-not','tag','legend')),0)
    return;
else
    error(['Either the figure contains multiple axes created without using the '...
           'default MATLAB function subplot.m, or this version of SPACEPLOTS is '...
           'not compatible with the default MATLAB '...
           'function subplot.m. In order to ensure compatibility, replace '...
           'the original subplot.m with the modified version available with '...
           'SPACEPLOTS v2.0'])
end

fPadLeft = fpad(1); fPadRight = fpad(2);
fPadTop = fpad(3);  fPadBottom = fpad(4);
axPadH = axpad(1); axPadV = axpad(2);

nRows = size(hAxGrid,1);
nCols = size(hAxGrid,2);


%% Get Current Axes Information

% hAxGrid is the grid of axes handles, where (1,1) corresponds to the
% bottom-left corner

TightInsets = cell(size(hAxGrid));

for i = 1:nRows
    for j = 1:nCols       
        if isequal(hAxGrid(i,j),0)    
            TightInsets{i,j} = zeros(1,4);             
        else           
            TightInsets{i,j} = get(hAxGrid(i,j),'TightInset');
        end
    end
end


%% Define New Properties

OuterPositions = cell(size(hAxGrid));
LooseInsets = cell(size(hAxGrid));

opWidth = (1-fPadLeft-fPadRight-((nCols-1)*axPadH))/nCols;
opHeight = (1-fPadTop-fPadBottom-((nRows-1)*axPadV))/nRows;

for i = 1:nRows
    for j = 1:nCols
        
        if isequal(hAxGrid(i,j),0), continue; end
        
        nop(1) = fPadLeft + (j-1)*(opWidth+axPadH);
        nop(2) = fPadBottom + (i-1)*(opHeight+axPadV);
        nop(3) = opWidth;
        nop(4) = opHeight;
        
        OuterPositions{i,j} = nop;
          
        inLeft = 0; inRight = 0; inTop = 0; inBottom = 0;
        for m = 1:nRows
            inLeft = max([inLeft TightInsets{m,j}(1)]);
            inRight = max([inRight TightInsets{m,j}(3)]);
        end
        for m = 1:nCols
            inTop = max([inTop TightInsets{i,m}(4)]);
            inBottom = max([inBottom TightInsets{i,m}(2)]);
        end
        
        LooseInsets{i,j} = [inLeft inBottom inRight inTop];
        
    end
end


%% Reposition Axes

moved = zeros(size(hAxGrid));

for i = 1:nRows
    for j = 1:nCols
        
        if isequal(hAxGrid(i,j),0), continue; end     
        if moved(i,j) == 1, continue; end
        
        ax = hAxGrid(i,j);

        inds = find(hAxGrid == ax);
        
        if numel(inds) == 1
            
            op = OuterPositions{i,j};
            li = LooseInsets{i,j};
            
            lb = [op(1)+li(1) op(2)+li(2) op(3)-li(1)-li(3) op(4)-li(2)-li(4)];
            
            set(ax,'OuterPosition',op)
            set(ax,'Position',lb)
          
            moved(i,j) = 1;
            
        elseif numel(inds) > 1
            
            r = zeros(numel(inds),1);
            c = zeros(numel(inds),1);
            
            for k = 1:numel(inds)
                [rr,cc] = ind2sub(size(hAxGrid),inds(k));
                r(k) = rr; c(k) = cc;
            end
            
            rmin = min(r); rmax = max(r);
            cmin = min(c); cmax = max(c);
            
            op(1) = OuterPositions{rmin,cmin}(1);
            op(2) = OuterPositions{rmin,cmin}(2);
            op(3) = OuterPositions{rmax,cmax}(1) + ...
                    OuterPositions{rmax,cmax}(3) - ...
                    OuterPositions{rmin,cmin}(1);
            op(4) = OuterPositions{rmax,cmax}(2) + ...
                    OuterPositions{rmax,cmax}(4) - ...
                    OuterPositions{rmin,cmin}(2);
            
            li(1) = LooseInsets{rmin,cmin}(1);
            li(2) = LooseInsets{rmin,cmin}(2);
            li(3) = LooseInsets{rmax,cmax}(3);
            li(4) = LooseInsets{rmax,cmax}(4);
            
            lb = [op(1)+li(1) op(2)+li(2) op(3)-li(1)-li(3) op(4)-li(2)-li(4)];
            
            set(ax,'OuterPosition',op)
            set(ax,'Position',lb)
           
            moved(inds) = 1;
            
        end
        
        set(ax,'ActivePositionProperty','OuterPosition')
        
    end
end

%% Reset Figure Properties

set(gcf,'units',u1)
setappdata(gcf,'SubplotGrid',hAxGrid)