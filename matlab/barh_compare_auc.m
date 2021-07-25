function [] = barh_compare_auc(pfile, pttl, lcs, varargin)
    %BARH_COMPARE_AUC
    %
    %   [] = BARH_COMPARE_AUC(pfile, pttl, lcs, varargin);
    %
    %       Makes a horizontal barplot of AUCs.
    %
    % Input
    % -----
    % [char]
    % pfile:    The filename of the plot.
    %           Note that the file extension should be either 'eps' or 'png'.
    %           default: 'png'
    %
    % [char]
    % pttl:     The plot title.
    %
    % [cell]
    % lcs:      Labels of x-axis and colors of bars. Must be the same number as
    %           varargin. I.e., assume varargin has n inputs, then lcs should of
    %           dimension n-by-2.
    %           {'label1', [color1]; ...
    %            'label2', [color2]; ...
    %            ...
    %            'labeln', [colorn]}
    %           where colors are 1-by-3 [r, g, b] tuples.
    %
    % [cell]
    % varargin: If varginin{i} is a char, it should be the filename of a
    %           4-column CSV, delimited by SPACE.
    %           <network> c<cutoff> d<d> <auc>
    %           See [../scripts/get-auc-batch.sh]
    %           Note that <network> in each file should be the same (string
    %           before the last -)
    %
    %           If varargin{i} is a string, it refers to the filename of a single AUC.
    %
    %           If varargin{i} is a 1-by-2 cell, it refers to two paired
    %           results, the first without ppi and the second with ppi.
    %
    % Output
    % ------
    % None.

    % check inputs {{{
    if nargin < 4
        error('barh_compare_auc:InputCount', 'Expected at least 4 inputs.');
    end

    % pfile
    validateattributes(pfile, {'char'}, {'nonempty'}, '', 'pfile', 1);
    [~, ~, e] = fileparts(pfile);
    if isempty(e)
        e = '.png';
    end
    ext = validatestring(e, {'.eps', '.png'}, '', 'pfile', 1);
    if strcmpi(ext, '.eps')
        device_op = '-depsc';
    elseif strcmpi(ext, '.png')
        device_op = '-dpng';
    end

    % pttl
    validateattributes(pttl, {'char'}, {}, '', 'pttl', 2);

    % lcs
    validateattributes(lcs, {'cell'}, {}, '', 'lcs', 3);
    labels = lcs(:,1)';
    colors = lcs(:,2)';
    n = size(lcs, 1);

    % varargin
    validateattributes(varargin, {'cell'}, {'numel', n}, '', 'varargin', 4);
    % }}}

    % load and parse varargin {{{
    aucs   = zeros(1, n);
    aucs2  = zeros(1, n);
    stds   = zeros(1, n);
    stds2  = zeros(1, n);
    paired = false(1, n);
    for i = 1 : n
        if iscell(varargin{i})
            if numel(varargin{i}) ~= 2
                msg = sprintf('Must be a pair of two results.');
                error('barh_compare_auc:InputErr %s', msg);
            end
            paired(i) = true;
            [aucs(i), stds(i)]   = loc_load_res(varargin{i}{1});
            [aucs2(i), stds2(i)] = loc_load_res(varargin{i}{2});
        else
            [aucs(i), stds(i)] = loc_load_res(varargin{i});
        end
    end

    % reverse up side down
    labels = flip(labels);
    colors = flip(colors);
    aucs   = flip(aucs);
    aucs2  = flip(aucs2);
    stds   = flip(stds);
    stds2  = flip(stds2);
    paired = flip(paired);
    % }}}

    % plot {{{
    h  = figure('Visible', 'off');
    ax = gca;
    box on;
    bw = 0.7;
    for i = 1 : n
        if ~paired(i)
            loc_draw_errbar(ax, i, bw, aucs(i), stds(i), colors{i});
        else
            c_dark = colors{i} / 3;
            % put the one with ppi below (smaller in Y-axis!)
            loc_draw_errbar(ax, i+bw/4, bw/2, aucs(i),  stds(i),  colors{i});
            loc_draw_errbar(ax, i-bw/4, bw/2, aucs2(i), stds2(i), c_dark);
        end
    end

    ax.Title.String  = pttl;
    % ax.FontSize      = 6;
    ax.XLim          = [0.5 0.8];
    ax.XLabel.String = 'AUC';
    ax.YTick         = 1:n;
    ax.YTickLabel    = labels;
    ax.YLim          = [0, n+1];

    % embed_canvas(h, 10, 8);
    embed_canvas(h, 5, 8);
    print(h, pfile, device_op, '-r300');
    close;
    % }}}
end

% function loc_load_res {{{
function [auc, stds] = loc_load_res(arg)
    if isnumeric(arg)
        auc  = arg;
        stds = 0;
    elseif ischar(arg)
        if ~exist(arg, 'file')
            msg = sprintf('Cannot open file [%s].', arg);
            error('barh_compare_auc:FileErr %s', msg);
        end
        % read data
        data = textscan(fopen(arg, 'r'), '%s%s%s%f', 'Delimiter', ' ');
        % netname = regexprep(data{1}, '-[0-9]+$', ''); % chop off the dangling #
        % if numel(unique(netname)) > 1
        %     msg = sprintf('More than one networks in [%s].', arg);
        %     error('barh_compare_auc:InputErr', msg);
        % end
        auc  = mean(data{4});
        stds = std(data{4});
    else
        msg = 'Required either a number or a file name.';
        error('barh_compare_auc:InputErr', msg);
    end
end
% }}}

% function loc_draw_errbar {{{
function [] = loc_draw_errbar(ax, x, width, height, err, clr)
    lw = 2;
    rectangle(ax, 'Position', [0, x-width/2, height, width], 'FaceColor', clr);
    if err > 0
        % draw an error bar
        line(ax, height+[-err/2, err/2], [x, x], 'Color', 'Black', 'LineWidth', lw);
        line(ax, repmat(height-err/2, 1, 2), [x-0.4*width, x+0.4*width], 'Color', 'Black', 'LineWidth', lw);
        line(ax, repmat(height+err/2, 1, 2), [x-0.4*width, x+0.4*width], 'Color', 'Black', 'LineWidth', lw);
    end
end
% }}}

% -------------
% Yuxiang Jiang (yuxjiang@indiana.edu)
% Department of Computer Science
% Indiana University, Bloomington
% Last modified: Thu 24 Jul 2021
