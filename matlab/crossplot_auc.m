function [] = crossplot_auc(pfile, pttl, R, P, C, ppi)
  %CROSSPLOT_AUC
  %
  % [] = CROSSPLOT_AUC(pfile, pttl, R, P, C, ppi);
  %
  %   Makes a cross-sectional plots for regions, periods and combinations of AUC comparing with PPI.
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
  % R:        4-by-1 cell array of AUCs: (each cell contains a 1-by-3 double
  %           array indicating AUC from [raw, ix ppi, ux ppi]
  %
  % [cell]
  % P:        1-by-12 cell array of AUCs: (each cell contains a 1-by-3 double
  %           array indicating AUC from [raw, ix ppi, ux ppi]
  %
  % [cell]
  % C:        4-by-12 cell array of AUCs: (each cell contains a 1-by-3 double
  %           array indicating AUC from [raw, ix ppi, ux ppi]
  %
  % [double]
  % ppi:      The auc of ppi network.
  %
  % Output
  % ------
  % None.

  % check inputs {{{
  if nargin ~= 6
    error('crossplot_auc:InputCount', 'Expected 6 inputs.');
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

  % R
  validateattributes(R, {'cell'}, {'numel', 4}, '', 'R', 3);

  % P
  validateattributes(P, {'cell'}, {'numel', 12}, '', 'P', 4);

  % C
  validateattributes(C, {'cell'}, {'nrows', 4, 'ncols', 12}, '', 'P', 5);

  % ppi
  validateattributes(ppi, {'double'}, {'numel', 1}, '', 'ppi', 6);
  % }}}

  % plot data {{{
  h = figure;
  axis off;
  ax = gca;

  L = 0.4; % minimum AUC
  H = 0.8; % maximum AUC

  % the base AUC to show relative perf (corresp. white color)
  % base = ppi;
  base = (L + H) / 2;
  scale = 3;

  for i = 0 : 4
    for j = 0 : 12
      if i == 0 && j == 0
        continue;
      end
      if j == 0
        drawIcon(ax, [-0.5 5 - i] * scale, R{i}, L, H, base);
        fprintf('R%d,%.3f,%.3f,%.3f\n', i, R{i}(1), R{i}(2), R{i}(3));
      elseif i == 0
        drawIcon(ax, [j, 5.5] * scale, P{j}, L, H, base);
        % fprintf('P%d,%.3f,%.3f,%.3f\n', j+1, P{j}(1), P{j}(2), P{j}(3));
      else
        drawIcon(ax, [j, 5 - i] * scale, C{i, j}, L, H, base);
        % fprintf('"R%d,P%d",%.3f,%.3f,%.3f\n', i, j+1, C{i,j}(1), C{i,j}(2), C{i,j}(3));
      end
    end
  end

  % label text {{{
  for i = 1 : 4
    text(ax, -1 * scale, (5 - i) * scale, sprintf('R%d', i), ...
      'HorizontalAlignment', 'right', ...
      'FontSize', 6);
  end

  for j = 1 : 12
    text(ax, j * scale, 6 * scale, sprintf('P%d', j + 1), ...
      'HorizontalAlignment', 'left', ...
      'Rotation', 90, ...
      'FontSize', 6);
  end
  % }}}

  % draw box {{{
  [~, ~, cb, cy] = pfp_rgby;
  bw = 0.5  * scale;
  bh = 5    * scale;
  bx = 14.5 * scale;
  steps = 30;
  anchor = (base - L) / (H - L) * bh + scale - 1; % the y position of base
  ppimark = (ppi - L) / (H - L) * bh + scale - 1;
  above = (H - base) / (H - L) * bh;
  below = (base - L) / (H - L) * bh;
  for i = 1 : steps
    dh = above / steps;
    auc = base + i * (H - base) / steps;
    c = cy + ([1 1 1] - cy) * (H - auc) / (H - base);
    rectangle(ax, 'Position', [bx anchor + dh * (i - 1) bw dh], ...
      'FaceColor', c, 'EdgeColor', c);

    dh = below / steps;
    auc = base - i * (base - L) / steps;
    c = cb + ([1 1 1] - cb) * (auc - L) / (base - L);
    rectangle(ax, 'Position', [bx anchor - dh * i bw dh], ...
      'FaceColor', c, 'EdgeColor', c);
  end
  text(ax, bx - bw * 2.5, anchor + above, sprintf('%.2f', H), ...
    'FontSize', 6, ...
    'VerticalAlign', 'middle');
  text(ax, bx + bw * 1.5, ppimark, sprintf('%.2f (ppi)', ppi), 'FontSize', 6);
  rectangle(ax, 'Position', [bx ppimark bw bh / 2 / steps], ...
    'FaceColor', 'k', 'EdgeColor', 'k');
  text(ax, bx - bw * 2.5, anchor - below, sprintf('%.2f', L), ...
    'FontSize', 6, ...
    'VerticalAlign', 'middle');
  % }}}

  ax.Title.String = pttl;
  ax.DataAspectRatio = [1 1 1];
  ax.DataAspectRatioMode = 'manual';
  embed_canvas(h, 5, 4);
  print(h, pfile, device_op, '-r600');
  close;
  % }}}
end

% function: drawIcon {{{
function [] = drawIcon(ax, shift, aucs, L, H, base)
  if any(isnan(aucs))
    return
  end
  r = 1;
  ec = [.5 .5 .5]; % edge color
  [~, ~, cb, cy] = pfp_rgby;
  for i = 1 : 3
    if aucs(i) - base >= 0
      alpha = (H - aucs(i)) / (H - base);
      alpha = max(0, min(.99, alpha));
      c = cy + ([1 1 1] - cy) * alpha;
    else
      alpha = (aucs(i) - L) / (base - L);
      alpha = max(0, min(.99, alpha));
      c = cb + ([1 1 1] - cb) * alpha;
    end
    a = 1 / 3 * 2 * pi * (i - 1) + pi / 6;
    b = 1 / 3 * 2 * pi * i + pi / 6;
    theta = linspace(a, b, 30);
    x = shift(1) + r .* [cos(b), 0, cos(a), cos(theta)];
    y = shift(2) + r .* [sin(b), 0, sin(a), sin(theta)];
    patch(ax, 'XData', x, 'YData', y, 'FaceColor', c, 'EdgeColor', ec, 'FaceAlpha', 0.8);
  end
end
% }}}

% -------------
% Yuxiang Jiang (yuxjiang@indiana.edu)
% Department of Computer Science
% Indiana University, Bloomington
% Last modified: Sun Dec  2 00:43:36 2018
