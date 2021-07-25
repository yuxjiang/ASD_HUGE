function [] = barh_compare_pvalue3(pfile, pttl, lcs, data, varargin)
  %BARH_COMPARE_PVALUE3
  %
  % [] = BARH_COMPARE_PVALUE3(pfile, pttl, lcs, data, varargin);
  %
  %   Makes a horizontal barplot of (tail) P-value comparisons.
  %
  % Input
  % -----
  % (required)
  % [char]
  % pfile:    The filename of the plot.
  %           Note that the file extension should be either 'eps' or 'png'.
  %           default: 'png'
  %
  % [char]
  % pttl:     The plot title.
  %
  % [cell]
  % lcs:      n-by-2 cell array of labels of x-axis and colors of bars. n Must
  %           be the same as the number of entries in 'data'.
  %
  % [cell]
  % data:     If data{i} is a char, it should be the filename of a
  %           single-column file with <pvalue> and the last row should be the single
  %           standalone run without -bootstrap (see ../script/tail-stat-bootstrap.sh
  %           for more details.)
  %
  %           If data{i} is a number, it refers to a single p-value.
  %
  %           If data{i} is a 1-by-3 cell, it refers to three bundled
  %           results,
  %           1. original
  %           2. intersection with ppi
  %           3. union with ppi
  %
  % (optional) Name-value pairs
  % [logical]
  % labeled:  The toggle for showing the label of each method
  %           default: true
  %
  % [double]
  % range:    The range for -log(p)
  %           default: 20
  %
  % Output
  % ------
  % None.

  % check inputs {{{
  if nargin < 4
    error('barh_compare_pvalue3:InputCount', 'Expected at least 4 inputs.');
  end

  % pfile
  validateattributes(pfile, {'char'}, {'nonempty'}, '', 'pfile', 1);
  [p, f, e] = fileparts(pfile);
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
  labels = lcs(:, 1)';
  colors = lcs(:, 2)';
  n = numel(labels);

  % data
  validateattributes(data, {'cell'}, {'numel', n}, '', 'data', 4);
  % }}}

  % check extra inputs {{{
  p = inputParser;
  addParameter(p, 'labeled', true, @(b)islogical(b));
  addParameter(p, 'range', 20, @(x)isnumeric(x));
  addParameter(p, 'fwidth', 4, @(x)isnumeric(x));
  addParameter(p, 'haslegend', false, @islogical);
  parse(p, varargin{:});
  % }}}

  % load and parse data {{{
  logp1 = zeros(1, n);
  logp2 = zeros(1, n);
  logp3 = zeros(1, n);

  stds1 = zeros(1, n);
  stds2 = zeros(1, n);
  stds3 = zeros(1, n);

  bundeled = false(1, n);
  for i = 1 : n
    if iscell(data{i})
      if numel(data{i}) ~= 3
        msg = sprintf('Must be a pair of three results.');
        error('barh_compare_pvalue3:InputErr', msg);
      end
      bundeled(i) = true;
      [logp1(i), stds1(i)] = loc_load_res(data{i}{1});
      [logp2(i), stds2(i)] = loc_load_res(data{i}{2});
      [logp3(i), stds3(i)] = loc_load_res(data{i}{3});
    else
      [logp1(i), stds1(i)] = loc_load_res(data{i});
    end
  end
  % reverse up side down
  labels   = flip(labels);
  colors   = flip(colors);
  logp1    = flip(logp1);
  logp2    = flip(logp2);
  logp3    = flip(logp3);
  stds1    = flip(stds1);
  stds2    = flip(stds2);
  stds3    = flip(stds3);
  bundeled = flip(bundeled);
  % }}}

  % plot {{{
  h  = figure('Visible', 'off');
  ax = gca;
  box on;
  barWidth = 0.8;
  for i = 1 : n
    if ~bundeled(i)
      loc_draw_errbar(ax, i, barWidth/3, logp1(i), stds1(i), colors{i});
    else
      c_dark   = colors{i} / 1.5;
      c_darker = colors{i} / 2;
      % put the one with ppi below (smaller in Y-axis!)
      loc_draw_errbar(ax, i+barWidth/3, barWidth/3, logp1(i), stds1(i), colors{i});
      loc_draw_errbar(ax, i,            barWidth/3, logp2(i), stds2(i), c_dark);
      loc_draw_errbar(ax, i-barWidth/3, barWidth/3, logp3(i), stds3(i), c_darker);
    end
  end

  ax.Title.String  = pttl;
  ax.XLim = [0 p.Results.range];
  ax.FontSize      = 14;
  ax.XLabel.String = '-log10(p-value)';
  ax.YTick         = 1 : n;
  if p.Results.labeled
    ax.YTickLabel = labels;
  else
    ax.YTickLabel = [];
  end
  ax.YLim          = [0, n+.5];

  % statistically significant line
  line(repmat(-log10(0.05), 1, 2), ax.YLim, 'LineStyle', ':', 'LineWidth', 2);

  % Bonferroni-corrected
  line(repmat(-log10(0.05 / numel(data)), 1, 2), ax.YLim, ...
    'LineStyle', ':', 'LineWidth', 1.8, 'Color', [.7 .7 .7]);

  if p.Results.haslegend
    loc_draw_legend(ax, 3.2, ax.YLim(2) - 0.7, colors{1});
  end

  fig_height = ceil(numel(data) / 3);
  embed_canvas(h, p.Results.fwidth, fig_height); % for periods/regions
  print(h, pfile, device_op, '-r600');
  close;
  % }}}
end

% function loc_load_res {{{
function [logp, stds] = loc_load_res(arg)
  if isnumeric(arg)
    logp = -log10(arg);
    stds = 0;
  elseif ischar(arg)
    if ~exist(arg, 'file')
      msg = sprintf('Cannot open file [%s].', arg);
      error('barh_compare_pvalue3:FileErr %s', msg);
    end
    % read data
    % data = textscan(fopen(arg, 'r'), '%s%s%s%f', 'Delimiter', ' ');
    % netname = regexprep(data{1}, '-[0-9]+$', ''); % chop off the dangling #
    % if numel(unique(netname)) > 1
    %   msg = sprintf('More than one networks in [%s].', arg);
    %   error('barh_compare_pvalue3:InputErr', msg);
    % end
    data = pfp_loaditem(arg, 'numeric');
    pvalues = -log10(data);
    % logp = mean(pvalues);
    logp = pvalues(end);
    stds = std(pvalues(1:end-1));
  else
    msg = 'Required either a number or a file name.';
    error('barh_compare_pvalue3:InputErr', msg);
  end
end
% }}}

% function loc_draw_errbar {{{
function [] = loc_draw_errbar(ax, x, width, height, err, clr)
  rectangle(ax, 'Position', [0, x-width/2, height, width], 'FaceColor', clr);
  if err > 0
    % draw an error bar
    line(ax, height+[-err/2, err/2], [x, x]);
    line(ax, repmat(height-err/2, 1, 2), [x-0.4*width, x+0.4*width]);
    line(ax, repmat(height+err/2, 1, 2), [x-0.4*width, x+0.4*width]);
  end
end
% }}}

function [] = loc_draw_legend(ax, x, y, clr)
  % ax.Position(3) = 0.6;
  skip = .8;
  c_dark   = clr / 1.5;
  c_darker = clr / 2;
  c = colormap('lines');

  rectangle(ax, 'Position', [x-0.1, y-5*skip, 2.5, 4.5], 'EdgeColor', 'w', 'FaceColor', 'w')

  loc_draw_rect_sample(ax, x, y, clr, 'GE');
  loc_draw_rect_sample(ax, x, y - skip, c_dark, 'GE $\cap$ PPI');
  loc_draw_rect_sample(ax, x, y - 2 * skip, c_darker, 'GE $\cup$ PPI');
  loc_draw_line_sample(ax, x, y - 3 * skip, c(1,:), '$p = 0.05$');
  loc_draw_line_sample(ax, x, y - 4 * skip, [.7 .7 .7], '$p^\prime = 0.05$');
end

function [] = loc_draw_rect_sample(ax, x, y, c, t)
  h = 0.4; w = 0.5;
  rectangle(ax, 'Position', [x, y - h, w, h], 'FaceColor', c);
  text(x+h+0.2, y-h/2, t, 'Interpreter', 'latex');
end

function [] = loc_draw_line_sample(ax, x, y, c, t)
  h = 0.4; w = 0.5;
  line(ax, [x, x+w], [y-h/2, y-h/2], 'Color', c, 'LineStyle', ':', 'LineWidth', 2);
  text(x+h+0.2, y-h/2, t, 'Interpreter', 'latex');
end

% -------------
% Yuxiang Jiang (yuxjiang@indiana.edu)
% Department of Computer Science
% Indiana University, Bloomington
% Last modified: Tue 13 Jul 2021 02:43:17 AM E