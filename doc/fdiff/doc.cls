%=====================================================================
% ray.cls
% Raytheon Class
%=====================================================================

%=====================================================================
% Identification
%=====================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{doc}[2017/01/01 Generic Document Class]

\LoadClass[letterpaper, 12pt]{article}

\RequirePackage[usenames,dvipsnames]{xcolor}
\RequirePackage{fancyhdr}
\RequirePackage[top=0.7in,bottom=0.7in,left=0.5in,right=0.5in]{geometry}
\RequirePackage{graphicx}
\RequirePackage{empheq}
\RequirePackage{ifthen}
\RequirePackage{lastpage}
\RequirePackage{listings}
\RequirePackage{math}
\RequirePackage{tocloft}


%=====================================================================
% Commands
%=====================================================================


\newcommand\classification[1]{\renewcommand\@classification{ \textsc{ {#1} } }}
\newcommand\@classification{ \textsc{ \textcolor{ForestGreen}{unclassified} } }

\newcommand\logo[1]{\renewcommand\@logo{ \includegraphics[height=17px]{#1}} }
\newcommand\logoraw[1]{\renewcommand\@logo{ #1 }}
\newcommand\@logo{ } 


\setlength{\headheight}{15pt}
\renewcommand{\headrulewidth}{1pt}
\renewcommand{\footrulewidth}{1pt}
\lhead{\@title}\chead{\@classification}\rhead{ \@logo }
\lfoot{\@author\ on \today}\cfoot{\@classification}\rfoot{Page\ \thepage\ out of \pageref{LastPage}}
\pagestyle{fancy}


\ifx\pdfoutput\undefined                         %LaTeX
  \RequirePackage[ps2pdf,bookmarks=true,hidelinks]{hyperref}
  \hypersetup{ %
    pdfauthor   = {\@author},
    pdftitle    = {\@title},
    pdfcreator  = {LaTeX with hyperref package},
    pdfproducer = {dvips + ps2pdf}
  }
\else                                            %PDFLaTeX
  \RequirePackage[pdftex,bookmarks=true,hidelinks]{hyperref}
  \hypersetup{ %
    pdfauthor   = {\@author},
    pdftitle    = {\@title},
    pdfcreator  = {LaTeX with hyperref package},
    pdfproducer = {dvips + ps2pdf}
  }
\pdfadjustspacing=1
\fi

%=====================================================================
% Configuration
%=====================================================================
\renewcommand*\thesection{\arabic{section}}

\setlength\parindent{0pt}

\allowdisplaybreaks

\def\nl{\\\hline}
