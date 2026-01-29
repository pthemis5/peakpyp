% thesis.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2025/10/16 Simple Thesis Class with A&A style]

% Load base class
\LoadClass[a4paper,12pt]{report}

% ---------------------------------------------------
% Packages and configuration
% ---------------------------------------------------
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{natbib}           % for A&A style citations
\RequirePackage{hyperref}         % clickable TOC and refs
\RequirePackage{geometry}
\geometry{margin=2.5cm}

% ---------------------------------------------------
% Bibliography and citation setup
% ---------------------------------------------------
% A&A uses natbib with specific options
\bibpunct{(}{)}{;}{a}{}{,}

% ---------------------------------------------------
% Title page definition
% ---------------------------------------------------
\renewcommand{\maketitle}{
  \begin{titlepage}
    \centering
    {\LARGE \bfseries \@title \par}
    \vspace{1.5cm}
    {\large \@author \par}
    \vfill
    {\large \@date \par}
  \end{titlepage}
}

% ---------------------------------------------------
% Table of contents
% ---------------------------------------------------
\newcommand{\makecontents}{
  \tableofcontents
  \clearpage
}
\newcommand*\aap{A\&A}
\let\astap=\aap
\newcommand*\aapr{A\&A~Rev.}
\newcommand*\aaps{A\&AS}
\newcommand*\actaa{Acta Astron.}
\newcommand*\aj{AJ}
\newcommand*\ao{Appl.~Opt.}
\let\applopt\ao
\newcommand*\apj{ApJ}
\newcommand*\apjl{ApJ}
\let\apjlett\apjl
\newcommand*\apjs{ApJS}
\let\apjsupp\apjs
\newcommand*\aplett{Astrophys.~Lett.}
\newcommand*\apspr{Astrophys.~Space~Phys.~Res.}
\newcommand*\apss{Ap\&SS}
\newcommand*\araa{ARA\&A}
\newcommand*\azh{AZh}
\newcommand*\baas{BAAS}
\newcommand*\bac{Bull. astr. Inst. Czechosl.}
\newcommand*\bain{Bull.~Astron.~Inst.~Netherlands}
\newcommand*\caa{Chinese Astron. Astrophys.}
\newcommand*\cjaa{Chinese J. Astron. Astrophys.}
\newcommand*\fcp{Fund.~Cosmic~Phys.}
\newcommand*\gca{Geochim.~Cosmochim.~Acta}
\newcommand*\grl{Geophys.~Res.~Lett.}
\newcommand*\iaucirc{IAU~Circ.}
\newcommand*\icarus{Icarus}
\newcommand*\jcap{J. Cosmology Astropart. Phys.}
\newcommand*\jcp{J.~Chem.~Phys.}
\newcommand*\jgr{J.~Geophys.~Res.}
\newcommand*\jqsrt{J.~Quant.~Spectr.~Rad.~Transf.}
\newcommand*\jrasc{JRASC}
\newcommand*\memras{MmRAS}
\newcommand*\memsai{Mem.~Soc.~Astron.~Italiana}
\newcommand*\mnras{MNRAS}
\newcommand*\na{New A}
\newcommand*\nar{New A Rev.}
\newcommand*\nat{Nature}
\newcommand*\nphysa{Nucl.~Phys.~A}
\newcommand*\pasa{PASA}
\newcommand*\pasj{PASJ}
\newcommand*\pasp{PASP}
\newcommand*\physrep{Phys.~Rep.}
\newcommand*\physscr{Phys.~Scr}
\newcommand*\planss{Planet.~Space~Sci.}
\newcommand*\pra{Phys.~Rev.~A}
\newcommand*\prb{Phys.~Rev.~B}
\newcommand*\prc{Phys.~Rev.~C}
\newcommand*\prd{Phys.~Rev.~D}
\newcommand*\pre{Phys.~Rev.~E}
\newcommand*\prl{Phys.~Rev.~Lett.}
\newcommand*\procspie{Proc.~SPIE}
\newcommand*\qjras{QJRAS}
\newcommand*\rmxaa{Rev. Mexicana Astron. Astrofis.}
\newcommand*\skytel{S\&T}
\newcommand*\solphys{Sol.~Phys.}
\newcommand*\sovast{Soviet~Ast.}
\newcommand*\ssr{Space~Sci.~Rev.}
\newcommand*\zap{ZAp}
\makeatletter
\newcommand{\frontmatter}{
  \cleardoublepage
  \pagenumbering{roman}
}
\newcommand{\mainmatter}{
  \cleardoublepage
  \pagenumbering{arabic}
}
\newcommand{\backmatter}{
  \cleardoublepage
}
\makeatother

% \RequirePackage{hyperref}

% \hypersetup{
%   colorlinks=true,
%   linkcolor=blue,
%   citecolor=blue,
%   urlcolor=blue
% }

% ---------------------------------------------------
% End of class
% ---------------------------------------------------
\endinput
