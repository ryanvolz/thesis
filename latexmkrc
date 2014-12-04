# use lualatex
$pdflatex = 'lualatex %O %S';
$pdf_mode = 1; $postscript_mode = $dvi_mode = 0;

# record input and output files for better dependency tracking
$recorder = 1;

# use make
$use_make_for_missing_files = 1;

# set pdf viewer to system default
$pdf_previewer = 'xdg-open %O %S';
