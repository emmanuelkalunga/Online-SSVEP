function theAxis = subplot(nrows, ncols, thisPlot, varargin)
%SUBPLOT Create axes in tiled positions.
%   H = SUBPLOT(m,n,p), or SUBPLOT(mnp), breaks the Figure window
%   into an m-by-n matrix of small axes, selects the p-th axes for
%   the current plot, and returns the axis handle.  The axes are
%   counted along the top row of the Figure window, then the second
%   row, etc.  For example,
%
%       SUBPLOT(2,1,1), PLOT(income)
%       SUBPLOT(2,1,2), PLOT(outgo)
%
%   plots income on the top half of the window and outgo on the
%   bottom half. If the CurrentAxes is nested in a uipanel the
%   panel is used as the parent for the subplot instead of the
%   current figure.
%
%   SUBPLOT(m,n,p), if the axis already exists, makes it current.
%   SUBPLOT(m,n,p,'replace'), if the axis already exists, deletes it and
%   creates a new axis.
%   SUBPLOT(m,n,p,'align') places the axes so that the plot boxes 
%   are aligned, but does not prevent the labels and ticks from 
%   overlapping.
%   SUBPLOT(m,n,P), where P is a vector, specifies an axes position
%   that covers all the subplot positions listed in P.
%   SUBPLOT(H), where H is an axis handle, is another way of making
%   an axis current for subsequent plotting commands.
%
%   SUBPLOT('position',[left bottom width height]) creates an
%   axis at the specified position in normalized coordinates (in
%   in the range from 0.0 to 1.0).
%
%   SUBPLOT(..., PROP1, VALUE1, PROP2, VALUE2, ...) sets the
%   specified property-value pairs on the subplot axis. To add the
%   subplot to a specific figure pass the figure handle as the
%   value for the 'Parent' property.
%
%   If a SUBPLOT specification causes a new axis to overlap an
%   existing axis, the existing axis is deleted - unless the position
%   of the new and existing axis are identical.  For example,
%   the statement SUBPLOT(1,2,1) deletes all existing axes overlapping
%   the left side of the Figure window and creates a new axis on that
%   side - unless there is an axes there with a position that exactly
%   matches the position of the new axes (and 'replace' was not specified),
%   in which case all other overlapping axes will be deleted and the
%   matching axes will become the current axes.
%
%   SUBPLOT(111) is an exception to the rules above, and is not
%   identical in behavior to SUBPLOT(1,1,1).  For reasons of backwards
%   compatibility, it is a special case of subplot which does not
%   immediately create an axes, but instead sets up the figure so that
%   the next graphics command executes CLF RESET in the figure
%   (deleting all children of the figure), and creates a new axes in
%   the default position.  This syntax does not return a handle, so it
%   is an error to specify a return argument.  The delayed CLF RESET
%   is accomplished by setting the figure's NextPlot to 'replace'.
%
%   Be aware when creating subplots from scripts that the Position
%   property of subplots is not finalized until either a drawnow
%   command is issued, or MATLAB returns to await a user command. 
%   That is, the value obtained for subplot i by the command 
%   get(h(i),'Position') will not be correct until the script
%   refreshes the plot or exits. 
%
%   See also  GCA, GCF, AXES, FIGURE, UIPANEL

%   SUBPLOT(m,n,p,H) when H is an axes will move H to the specified
%   position. 
%   SUBPLOT(m,n,p,H,PROP1,VALUE1,...) will move H and apply the
%   specified property-value pairs 
%   
%   SUBPLOT(m,n,p) for non-integer p places the subplot at the
%   fraction p-floor(p) between the positions floor(p) and ceil(p)

%   Copyright 1984-2007 The MathWorks, Inc.
%   $Revision: 5.22.4.23 $  $Date: 2007/10/15 22:53:56 $

% we will kill all overlapping axes siblings if we encounter the mnp
% or m,n,p specifier (excluding '111').
% But if we get the 'position' or H specifier, we won't check for and
% delete overlapping siblings:
narg = nargin;
kill_siblings = 0;
create_axis = true;
move_axis = false;
delay_destroy = false;
useAutoLayout = true;
tol = sqrt(eps);
parent = get(0,'CurrentFigure');
parentfigure = parent;
if ~isempty(parent) && ~isempty(get(parent,'CurrentAxes'))
    parent = get(get(parent,'CurrentAxes'),'Parent');
    parentfigure = parent;
    if ~strcmp(get(parentfigure,'Type'),'figure')
        parentfigure = ancestor(parent,'figure');
    end
end
pvpairs = {};
preventMove = false;
% This is the percent offset from the subplot grid of the plotbox.
inset = [.2 .18 .04 .1]; % [left bottom right top]

if narg == 0 % make compatible with 3.5, i.e. subplot == subplot(111)
    nrows = 111;
    narg = 1;
end

%check for encoded format
handle = '';
position = '';
explicitParent = false;
explicitPosition = false;

if narg == 1
    % The argument could be one of 3 things:
    % 1) a 3-digit number 100 < num < 1000, of the format mnp
    % 2) a 3-character string containing a number as above
    % 3) an axis handle
    code = nrows;

    % turn string into a number:
    if(ischar(code))
        code = str2double(code);
    end
    
    % Check for NaN and Inf.
    if (~isfinite(code))
        error(id('SubplotIndexNonFinite'),'Index must be a finite 3-digit number of the format mnp.')
    end

    % number with a fractional part can only be an identifier:
    if(rem(code,1) > 0)
        handle = code;
        if ~strcmp(get(handle,'type'),'axes')
            error(id('InvalidAxesHandle'),'Requires valid axes handle for input.')
        end
        create_axis = false;
        % all other numbers will be converted to mnp format:
    else
        % Check for input out of range
        if (code <= 100 || code >= 1000)
            error(id('SubplotIndexOutOfRange'),'Index must be a 3-digit number of the format mnp.')
        end
        
        thisPlot = rem(code, 10);
        ncols = rem( fix(code-thisPlot)/10,10);
        nrows = fix(code/100);
        if nrows*ncols < thisPlot
            error(id('SubplotIndexTooLarge'),'Index exceeds number of subplots.');
        end
        kill_siblings = 1;
        if(code == 111)
            create_axis   = false;
            delay_destroy = true;
        else
            create_axis   = true;
            delay_destroy = false;
        end
    end

elseif narg == 2
    % The arguments MUST be the string 'position' and a 4-element vector:
    if(strcmpi(nrows, 'position'))
        pos_size = size(ncols);
        if(pos_size(1) * pos_size(2) == 4)
            position = ncols;
            explicitPosition = true;
        else
            error(id('InvalidPositionParameter'),...
                'Position must be of the form [left bottom width height].')
        end
    else
        error(id('UnknownOption'),'Unknown command option.')
    end
    kill_siblings = 1; % Kill overlaps here also.
    useAutoLayout = false;

elseif narg == 3
    % passed in subplot(m,n,p) -- we should kill overlaps
    % here too:
    kill_siblings = 1;

elseif narg >= 4
    if ~ischar(nrows)
        arg = varargin{1};
        if ~ischar(arg)
            % passed in subplot(m,n,p,H,...)
            handle = arg;
            parent = get(handle,'Parent');
            parentfigure = ancestor(handle,'figure');
            % If the parent is passed in explicitly, don't create a new figure
            % when the "NextPlot" property is set to "new" in the figure.
            explicitParent = true;
            set(parentfigure,'CurrentAxes',handle);
            move_axis = true;
            create_axis = false;
            if narg >= 5 && strcmpi(varargin{2},'PreventMove')
                preventMove = true;
                pvpairs = varargin(3:end);
            else
                pvpairs = varargin(2:end);
            end
        elseif strncmpi(arg,'replace',1)
            % passed in subplot(m,n,p,'replace')
            kill_siblings = 2; % kill nomatter what
        elseif strcmpi(arg,'align')
            % passed in subplot(m,n,p,'align')
            % since obeying position will remove the axes from the grid just set
            % useAutoLayout to false to skip adding it to the grid to start with
            useAutoLayout = false;
            kill_siblings = 1; % kill if it overlaps stuff
        elseif strcmpi(arg,'v6')
            % passed in subplot(m,n,p,'v6')
            % since obeying position will remove the axes from the grid just set
            % useAutoLayout to false to skip adding it to the grid to start with
            warning(['MATLAB:', mfilename, ':DeprecatedV6Argument'],...
                ['The ''v6'' argument to %s is deprecated,',...
                ' and will no longer be supported in a future release.'], upper(mfilename));
            useAutoLayout = false;
            kill_siblings = 1; % kill if it overlaps stuff
        else
            % passed in prop-value pairs
            kill_siblings = 1;
            pvpairs = varargin;
            par = find(strncmpi('Parent',pvpairs(1:2:end),6));
            if any(par)
                % If the parent is passed in explicitly, don't create a new figure
                % when the "NextPlot" property is set to "new" in the figure.
                explicitParent = true;
                parent = varargin{2*par(1)};
                parentfigure = ancestor(parent,'figure');
            end
        end
    else
        % Passed in "Position" syntax with P/V pairs
        % The arguments MUST be the string 'position' and a 4-element vector:
        if(strcmpi(nrows, 'position'))
            pos_size = size(ncols);
            if(pos_size(1) * pos_size(2) == 4)
                position = ncols;
                explicitPosition = true;
            else
                error(id('InvalidPositionParameter'),...
                    'Position must be of the form [left bottom width height].')
            end
        else
            error(id('UnknownOption'),'Unknown command option.')
        end
        kill_siblings = 1; % Kill overlaps here also.
        useAutoLayout = false;
        pvpairs = [{thisPlot} varargin];
        par = find(strncmpi('Parent',pvpairs(1:2:end),6));
        if any(par)
            % If the parent is passed in explicitly, don't create a new figure
            % when the "NextPlot" property is set to "new" in the figure.
            explicitParent = true;
            parent = pvpairs{2*par(1)};
            parentfigure = ancestor(parent,'figure');
        end
    end
end

% if we recovered an identifier earlier, use it:
if ~isempty(handle) && ~move_axis
    parent = get(handle,'Parent');
    parentfigure = ancestor(handle,'figure');
    set(parentfigure,'CurrentAxes',handle);
else  % if we haven't recovered position yet, generate it from mnp info:
    if isempty(parent),
        parent = gcf;
        parentfigure = parent;
    end
    if isempty(position)
        if min(thisPlot) < 1
            error(id('SubplotIndexTooSmall'),'Illegal plot number.')
        elseif max(thisPlot) > ncols*nrows
            error(id('SubplotIndexTooLarge'),'Index exceeds number of subplots.');
        else

            row = (nrows-1) -fix((thisPlot-1)/ncols);
            col = rem (thisPlot-1, ncols);

            % get default axis position in normalized units
            % If we have checked this quanitity once, cache it.
            if ~isappdata(parentfigure,'SubplotDefaultAxesLocation')
                if ~strcmp(get(parentfigure,'defaultaxesunits'),'normalized')
                    tmp = axes;
                    set(tmp,'units','normalized')
                    def_pos = get(tmp,'position');
                    delete(tmp)
                else
                    def_pos = get(parentfigure,'DefaultAxesPosition');
                end
                setappdata(parentfigure,'SubplotDefaultAxesLocation',def_pos);
            else
                def_pos = getappdata(parentfigure,'SubplotDefaultAxesLocation');
            end

            % compute outerposition and insets relative to figure bounds
            rw = max(row)-min(row)+1;
            cw = max(col)-min(col)+1;
            width = def_pos(3)/(ncols - inset(1) - inset(3));
            height = def_pos(4)/(nrows - inset(2) - inset(4));
            inset = inset.*[width height width height];
            outerpos = [def_pos(1) + min(col)*width - inset(1), ...
                def_pos(2) + min(row)*height - inset(2), ...
                width*cw height*rw];

            % adjust outerpos and insets for axes around the outside edges
            if min(col) == 0
                inset(1) = def_pos(1);
                outerpos(3) = outerpos(1)+outerpos(3);
                outerpos(1) = 0;
            end
            if min(row) == 0,
                inset(2) = def_pos(2);
                outerpos(4) = outerpos(2)+outerpos(4);
                outerpos(2) = 0;
            end
            if max(col) == ncols-1,
                inset(3) = max(0,1-def_pos(1)-def_pos(3));
                outerpos(3) = 1-outerpos(1);
            end
            if max(row) == nrows-1,
                inset(4) = max(0,1-def_pos(2)-def_pos(4));
                outerpos(4) = 1-outerpos(2);
            end

            % compute inner position
            position = [outerpos(1:2) + inset(1:2),...
                outerpos(3:4) - inset(1:2) - inset(3:4)];

        end
    end
end

% kill overlapping siblings if mnp specifier was used:
nextstate = get(parentfigure,'nextplot');

if strncmp(nextstate,'replace',7)
    nextstate = 'add';
elseif strncmp(nextstate,'new',3)
    kill_siblings = 0;
end

if(kill_siblings)
    if delay_destroy
        if nargout
            error(id('TooManyOutputs'),...
                'Function called with too many output arguments.')
        else
            set(parentfigure,'NextPlot','replace');
            return
        end
    end
    sibs = datachildren(parent);
    newcurrent = [];
    for i = 1:length(sibs)
        % Be aware that handles in this list might be destroyed before
        % we get to them, because of other objects' DeleteFcn callbacks...
        if(ishandle(sibs(i)) && strcmp(get(sibs(i),'Type'),'axes'))
            units = get(sibs(i),'Units');
            sibpos = get(sibs(i),'Position');
            if ~strcmp(units,'normalized')
                sibpos = hgconvertunits(parentfigure,sibpos,units,'normalized',parent);
            end
            intersect = 1;
            if((position(1) >= sibpos(1) + sibpos(3)-tol) || ...
                    (sibpos(1) >= position(1) + position(3)-tol) || ...
                    (position(2) >= sibpos(2) + sibpos(4)-tol) || ...
                    (sibpos(2) >= position(2) + position(4)-tol))
                intersect = 0;
            end
            if intersect
                % position is the proposed position of an axes, and
                % sibpos is the current position of an existing axes.
                % Since the bounding boxes of position and sibpos overlap,
                % we must determine whether to delete the sibling sibs(i)
                % whose normalized position is sibpos.
                
                % First of all, we check whether we must kill the sibling
                % "no matter what."
                if (kill_siblings == 2)
                    delete(sibs(i));
                    
                % If the proposed and existing axes overlap exactly, we do
                % not kill the sibling.  Rather we shall ensure later that
                % this sibling axes is set as the 'CurrentAxes' of its
                % parentfigure.
                    
                % Next we check for a partial overlap.
                elseif (any(abs(sibpos - position) > tol))
                    % The proposed and existing axes partially overlap. 
                    % Since the proposed and existing axes could each be
                    % "grid-generated" or "explicitly-specified", we must
                    % consider four possibilities for the overlap of
                    % "proposed" vs. "existing", i.e.
                    % (1) "grid-generated" vs. "grid-generated"
                    % (2) "grid-generated" vs. "explicitly-specified"
                    % (3) "explicitly-specified" vs. "grid-generated"
                    % (4) "explicitly-specified" vs. "explicitly-specified"
                    
                    % If the position of the proposed axes is
                    % "explicitly-specified", then the only condition that
                    % avoids killing the sibling is an exact overlap.
                    % However, we know that the overlap is partial.
                    if (explicitPosition)
                        delete(sibs(i));
                    else
                        % We know that the position of the proposed axes is
                        % "grid-generated".
                        
                        grid = getappdata(parent,'SubplotGrid');
                        % The SubplotGrid maintains an array of axes
                        % handles, one per grid location.  Axes that span
                        % multiple grid locations do not store handles in
                        % the SubplotGrid.
                        
                        if isempty(grid) || ~any(grid(:) == sibs(i)) || ...
                                size(grid,1) ~= nrows || size(grid,2) ~= ncols || ...
                                ~isscalar(row) || ~isscalar(col)
                            % If the sibling cannot be found in the grid, we
                            % kill the sibling.  Otherwise, the proposed and 
                            % existing axes are "grid-generated".  If we
                            % are changing the size of the grid, we kill
                            % the sibling.  Otherwise, "thisPlot" may be a
                            % vector of multiple grid locations, which
                            % causes a partial overlap between the proposed
                            % and existing axes, so we kill the sibling.
                            
                            % This check recognizes that there may be
                            % labels, colorbars, legends, etc. attached to
                            % the existing axes that have affected its
                            % position.  In such a case, we do not kill the
                            % sibling.
                            delete(sibs(i));
                        end
                    end
                end
                if ishandle(sibs(i))
                    % if this axes overlaps the other one exactly then
                    if ~isempty(newcurrent) && ishandle(newcurrent)
                        delete(newcurrent);
                    end
                    newcurrent = sibs(i);
                end
            end
        end
    end
    if ~isempty(newcurrent) && ishandle(newcurrent)
        set(parentfigure,'CurrentAxes',newcurrent);
        create_axis = false;
    end
    set(parentfigure,'NextPlot',nextstate);
end

% create the axis:
if create_axis
    if strcmp(nextstate,'new') && ~explicitParent
        parent = figure;
        parentfigure = parent;
    end
   ax = axes('units','normal','Position',position,...
             'LooseInset',inset,'Parent',parent,pvpairs{:});
  % TODO: Get axes to accept position args on command line
  set(ax,'units',get(double(parentfigure),'defaultaxesunits'))
    if useAutoLayout
        addAxesToGrid(ax,nrows,ncols,row,col,position);
    end
elseif move_axis && ~preventMove
    ax = handle;
    units = get(handle,'units');
    set(handle,'units','normal','Position',position,...
        'LooseInset',inset,'Parent',parent,pvpairs{:});
    set(handle,'units',units);
    if useAutoLayout
        addAxesToGrid(ax,nrows,ncols,row,col,position);
    end
else
    % this should only happen with subplot(H)
    ax = get(parentfigure,'CurrentAxes');
end

% return identifier, if requested:
if(nargout > 0)
    theAxis = ax;
end

% Create subplot listeners to align plot boxes automatically
function createListeners(p,axlisth)
setappdata(p,'SubplotListeners',[])
fig = p;
if ~isequal(get(fig,'Type'),'figure')
    fig = ancestor(fig,'figure');
end
list = [...
    handle.listener(axlisth,findprop(axlisth(1),'Units'), ... % must be first
    'PropertyPostSet',@axesUnitsPostSet);
    handle.listener(axlisth,findprop(axlisth(1),'Units'), ...
    'PropertyPreSet',@axesUnitsPreSet);
    handle.listener(axlisth,findprop(axlisth(1),'Position'), ...
    'PropertyPostSet',@axesMoved);
    handle.listener(axlisth,findprop(axlisth(1),'ActivePositionProperty'), ...
    'PropertyPreSet',@axesMoved);
    handle.listener(axlisth,findprop(axlisth(1),'Parent'), ...
    'PropertyPreSet',@axesMoved);
    handle.listener(axlisth,'AxisInvalidEvent',{@subplotlayoutInvalid,p});
    handle.listener(handle(fig),'FigureUpdateEvent',{@subplotlayout,p})];
for k=1:length(axlisth)
    ax = axlisth(k);
    if ~isappdata(double(ax),'SubplotDeleteListener')
        setappdata(double(ax),'SubplotDeleteListener',...
            handle.listener(ax,'ObjectBeingDestroyed', ...
            @axesDestroyed));
    end
end
setappdata(p,'SubplotListeners',list)

% Add ax to a matrix of handles in the specified location.
% The grid is stored on the parent appdata.
% Also store the insets in ax appdata.
% Only stores the axes if it is in a 1-by-1 cell and
% the grid size matches any existing grid.
function addAxesToGrid(ax,nrows,ncols,row,col,position)
p = get(ax,'parent');
grid = getappdata(p,'SubplotGrid');
if isempty(grid)
    grid = zeros(nrows,ncols);
end
if any(size(grid) ~= [nrows ncols]), return; end
if grid(row+1,col+1) == ax, return, end

% <SPACEPLOTS> Modifications made for compatibility with Spaceplots

if ~isequal(round(row),row) || ~isequal(round(col),col), return; end
setappdata(p,'spaceplots',1)

% In order to revert to original subplot file, delete the above
% modifications, and uncomment the following 2 lines:

% if length(row) ~= 1 || length(col) ~= 1, return; end
% if round(row) ~= row || round(col) ~= col, return; end

% <end SPACEPLOTS modifications>

grid(row+1,col+1) = ax;
list = grid(:);
list(list == 0) = []; % remove root
list(~ishandle(list)) = []; % remove invalid handles
createListeners(p,handle(list));
setappdata(p,'SubplotGrid',grid)
setappdata(ax,'SubplotPosition',position); % normalized
subplotlayoutInvalid(handle(ax),[],p);

% Remove ax from grid of subplots in p
function removeAxesFromGrid(p,ax)
grid = getappdata(p,'SubplotGrid');
if ~isempty(grid)
    n = grid == ax;
    if any(n(:))
        grid(n) = 0;
        list = grid(:);
        list(list == 0) = []; % remove root
        list(~ishandle(list)) = [];
        if isempty(list)
            rmappdata(p,'SubplotListeners');
            rmappdata(p,'SubplotGrid');
        else
            setappdata(p,'SubplotGrid',grid);
        end
    end
end

% Callback when axis moves to remove it from subplot layout grid
function axesMoved(hSrc,evdata) %#ok
ax = double(evdata.affectedObject);
% If the legend or colorbar is causing the move, do not remove the axes
% from the subplot grid. Do, however, update it's cached position:
if isappdata(ax,'inLayout') && ~isempty(getappdata(ax,'inLayout'))
    setappdata(ax,'SubplotPosition',get(ax,'Position'));
else
    removeAxesFromGrid(get(ax,'Parent'),ax);
end

% Callback when axis changes units
function axesUnitsPreSet(hSrc,evdata) %#ok
ax = double(evdata.affectedObject);
p = get(ax,'Parent');
list = getappdata(p,'SubplotListeners');
if ~isempty(list)
    set(list(2:end),'enable','off');
end

% Callback when axis done changing units
function axesUnitsPostSet(hSrc,evdata) %#ok
ax = double(evdata.affectedObject);
p = get(ax,'Parent');
list = getappdata(p,'SubplotListeners');
if ~isempty(list)
    set(list(2:end),'enable','on');
end

% Callback when axis destroyed
function axesDestroyed(hSrc,evdata) %#ok
ax = double(hSrc);
p = get(ax,'Parent');
if strcmp(get(p,'BeingDeleted'),'off')
    removeAxesFromGrid(p,ax);
elseif isappdata(p,'SubplotListeners')
    rmappdata(p,'SubplotListeners');
    rmappdata(p,'SubplotGrid');
end

function str = id(str)
str = ['MATLAB:subplot:' str];
