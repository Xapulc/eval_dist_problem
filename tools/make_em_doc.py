"""Генератор документа EM_один_тест.tex: числа берутся из data/B_rel_s24.  Запуск из корня: python3 tools/make_em_doc.py, затем xelatex."""
import os, sys, json, numpy as np, pandas as pd
from scipy.optimize import brentq
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
d = f"{root}/data/B_rel_s24"; prm = json.load(open(f"{d}/hidden/params.json")); p_true, s_true = np.array(prm["p"]), np.array(prm["s"])
X = pd.read_csv(f"{d}/public/dataset_train.csv").sort_values(["release_id", "eval_id", "run_no"]).red_flag.values.reshape(70, 40, 5).sum(2)
N, M = X.shape
nplus = (X > 0).sum(0); S = X.sum(0)


def mle(k):
    npl, s_ = int((k > 0).sum()), int(k.sum()); n = len(k)
    if npl == 0:
        return 0.0, float("nan"), npl, s_
    m = s_ / npl
    if m <= 1 + 1e-12:
        return 1.0, 0.0, npl, s_                              # boundary: s -> 0, p -> 1
    if m >= 5 - 1e-12:
        s = 1.0
    else:
        s = brentq(lambda t: 5 * t / (1 - (1 - t) ** 5) - m, 1e-9, 1 - 1e-12)
    q = npl / n; p = min(q / (1 - (1 - s) ** 5), 1.0)
    return p, s, npl, s_


def em(k, iters=500):
    npl, s_ = (k > 0).sum(), k.sum(); n = len(k)
    if npl == 0:
        return 0.0, float("nan"), 0
    p, s = npl / n, min(s_ / (5 * npl), 0.999)
    for it in range(iters):
        z = (1 - s) ** 5; w0 = p * z / (1 - p + p * z); W = npl + (n - npl) * w0
        p_new, s_new = W / n, s_ / (5 * W)
        if abs(p_new - p) < 1e-10 and abs(s_new - s) < 1e-10:
            p, s = p_new, s_new; break
        p, s = p_new, s_new
    return min(p, 1.0), s, it + 1


rows = [(j, *mle(X[:, j]), *em(X[:, j]), p_true[j], s_true[j]) for j in range(M)]
deg = [r for r in rows if r[3] > 0 and r[4] == r[3]]            # all positive K equal 1
zero = [r for r in rows if r[3] == 0]
print(f"tests with n+ = 0: {len(zero)}; tests with S = n+ (MLE p=1, s=0): {len(deg)}; median n+: {np.median(nplus)}")
# illustrative tests: a typical one, a degenerate one, a zero one, and a frequent one
typ = min((r for r in rows if r[3] >= 3 and r[4] > r[3]), key=lambda r: abs(r[3] - 4))
small = max((r for r in rows if 1 <= r[3] <= 2 and r[4] > r[3]), key=lambda r: r[1] / max(r[8], 1e-9))   # p-hat overshoots the truth most
freq = max(rows, key=lambda r: r[3])
pick = [typ, small, zero[0], freq]
# how often the degenerate case (all positive K equal 1 for some test) occurs in histories of this world
from evals_task.model import draw
cnt_hist, cnt_tests = 0, 0
for h in range(200):
    Xh = draw(np.random.default_rng(400000 + h), p_true, s_true, N)[1].sum(2); npl_h, S_h = (Xh > 0).sum(0), Xh.sum(0)
    dg = ((npl_h >= 1) & (S_h == npl_h)).sum(); cnt_tests += dg; cnt_hist += dg > 0
print(f"degenerate tests: histories with >=1: {cnt_hist/200:.2f}, mean per history {cnt_tests/200:.2f}")
def fmt(x, nd=3):
    return "---" if (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
tabrows = "\n".join(f"{r[0]} & {r[3]} & {r[4]} & {fmt(r[1])} & {fmt(r[2])} & {fmt(r[5])} & {fmt(r[6])} & {r[7]} & {fmt(r[8])} & {fmt(r[9])} \\\\" for r in pick)
# agreement EM vs closed form over all tests with n+ > 0
dif = max(max(abs(r[1] - r[5]), abs((r[2] if not np.isnan(r[2]) else 0) - (r[6] if not np.isnan(r[6]) else 0))) for r in rows if r[3] > 0)
print("max |EM - closed form| over tests:", dif)

tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{PT Serif}[Ligatures=TeX]
\setsansfont{Helvetica Neue}[Ligatures=TeX]
\setmonofont{Menlo}
\usepackage{polyglossia}
\setdefaultlanguage{russian}
\setotherlanguage{english}
\newfontfamily\cyrillicfont{PT Serif}[Ligatures=TeX]
\newfontfamily\cyrillicfontsf{Helvetica Neue}
\newfontfamily\cyrillicfonttt{Menlo}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{geometry}
\geometry{margin=22mm}
\usepackage{booktabs}
\usepackage{enumitem}
\setlist{nosep,leftmargin=*}
\usepackage{microtype}
\usepackage{hyperref}
\hypersetup{colorlinks,linkcolor=black,urlcolor=blue!50!black}
\renewcommand{\arraystretch}{1.12}
\emergencystretch=2em
\newcommand{\E}{\mathbf{E}}
\newcommand{\Prob}{\mathbf{P}}
\newcommand{\D}{\mathbf{D}}
\newcommand{\Ind}{\mathrm{I}}
\newcommand{\Bern}{\mathrm{Bern}}
\newcommand{\Bin}{\mathrm{Bin}}
\DeclareMathOperator*{\argmax}{arg\,max}
\theoremstyle{plain}
\newtheorem{theorem}{Теорема}
\newtheorem{lemma}{Лемма}
\newtheorem{proposition}{Утверждение}
\theoremstyle{definition}
\newtheorem{remark}{Замечание}

\title{Оценка параметров одного LLM-теста:\\ функция правдоподобия, ОМП и EM-алгоритм}
\author{к задаче «Экономия на эвалах»}
\date{25 октября 2026}

\begin{document}
\sloppy
\maketitle

\paragraph{Об обозначениях.} Используются обозначения, принятые в курсе математической статистики мехмата: выборка $X=(X_1,\dots,X_N)$, параметрическое семейство $\{\Prob_\theta,\ \theta\in\Theta\}$, функция правдоподобия $L(\theta;X)$, оценка максимального правдоподобия $\hat\theta$, математическое ожидание и дисперсия $\E_\theta$, $\D_\theta$, индикатор $\Ind\{A\}$, биномиальное распределение $\Bin(n,s)$, распределение Бернулли $\Bern(p)$. Конспекты кафедры на момент написания были недоступны (teach-in.ru и math.msu.ru не отвечали), поэтому сверить обозначения буквально не удалось.

\section{Модель для одного теста}

Фиксируем тест $j$ и далее индекс $j$ опускаем. Релизов $N=70$, на каждом тест прогнан $r=5$ раз. Скрытая величина $I_i\in\{0,1\}$ --- есть ли в релизе $i$ проблема в сценарии теста; наблюдаемая $K_i\in\{0,\dots,r\}$ --- число красных прогонов. Модель:
\[
I_1,\dots,I_N\ \text{независимы},\quad I_i\sim\Bern(p);\qquad K_i\mid I_i=1\ \sim\ \Bin(r,s),\qquad K_i\mid I_i=0\ \equiv 0 .
\]
Параметр $\theta=(p,s)\in\Theta=[0,1]^2$. Ложных красных нет: при $I_i=0$ все прогоны зелёные.

\begin{remark}
В данных записан порядок прогонов внутри релиза, но при $I_i=1$ прогоны независимы и одинаково распределены, поэтому вектор исходов пяти прогонов зависит от $\theta$ только через $K_i$: число красных --- достаточная статистика для одного релиза. Правдоподобие по исходным данным отличается от правдоподобия по $K_i$ множителем, не зависящим от $\theta$.
\end{remark}

Распределение одного наблюдения:
\begin{equation}\label{eq:pmf}
\Prob_\theta(K=0)=1-p+p(1-s)^r,\qquad
\Prob_\theta(K=k)=p\binom rk s^k(1-s)^{r-k},\quad k=1,\dots,r .
\end{equation}
Нуль возникает двумя путями: проблемы нет, либо она есть, но ни разу не проявилась.

\section{Функция правдоподобия и ОМП}

Выборка $X=(K_1,\dots,K_N)$. Обозначим
\[
n_+=\sum_{i=1}^N\Ind\{K_i\ge1\},\qquad n_0=N-n_+,\qquad S=\sum_{i=1}^NK_i .
\]
По \eqref{eq:pmf} функция правдоподобия равна
\begin{align}
L(\theta;X)&=\bigl[1-p+p(1-s)^r\bigr]^{n_0}\prod_{i:\,K_i\ge1}p\binom r{K_i}s^{K_i}(1-s)^{r-K_i}\notag\\
&=\bigl[1-p+p(1-s)^r\bigr]^{n_0}\,p^{\,n_+}\,s^{S}(1-s)^{rn_+-S}\prod_{i:\,K_i\ge1}\binom r{K_i},\label{eq:lik}
\end{align}
а логарифмическая функция правдоподобия, с точностью до слагаемого, не зависящего от $\theta$, ---
\begin{equation}\label{eq:loglik}
\ell(p,s)=n_0\log\bigl[1-p+p(1-s)^r\bigr]+n_+\log p+S\log s+(rn_+-S)\log(1-s).
\end{equation}
По теореме факторизации $(n_+,S)$ --- достаточная статистика: сколько релизов с красным и сколько всего красных.

\begin{lemma}[разделение параметров]\label{lem:split}
Положим $q=\Prob_\theta(K\ge1)=p\bigl(1-(1-s)^r\bigr)$. Отображение $(p,s)\mapsto(q,s)$ взаимно однозначно на $(0,1]\times(0,1]$, и
\begin{equation}\label{eq:split}
\ell=\underbrace{n_+\log q+n_0\log(1-q)}_{\ell_1(q)}\ +\ \underbrace{S\log s+(rn_+-S)\log(1-s)-n_+\log\bigl(1-(1-s)^r\bigr)}_{\ell_2(s)} .
\end{equation}
\end{lemma}

\begin{proof}
$1-q=1-p+p(1-s)^r$, а $p=q/(1-(1-s)^r)$; подставляем в \eqref{eq:loglik}.
\end{proof}

Второе слагаемое $\ell_2(s)$ --- логарифм правдоподобия выборки $\{K_i:\ K_i\ge1\}$ из \emph{усечённого в нуле} биномиального распределения $\Prob(K=k\mid K\ge1)=\binom rk s^k(1-s)^{r-k}/(1-(1-s)^r)$.

\begin{proposition}[ОМП в явном виде]\label{prop:mle}
Пусть $n_+\ge1$ и $\bar K_+=S/n_+$ --- среднее число красных по релизам с красным. Тогда
\begin{equation}\label{eq:mle}
\hat q=\frac{n_+}N,\qquad
\hat s:\quad \frac{r\hat s}{1-(1-\hat s)^r}=\bar K_+,\qquad
\hat p=\min\Bigl\{\frac{\hat q}{1-(1-\hat s)^r},\,1\Bigr\}.
\end{equation}
Уравнение для $\hat s$ имеет единственный корень на $(0,1)$ при $1<\bar K_+<r$. При $\bar K_+=1$ оценка вырождается: $\hat s\to0$, $\hat p=1$; при $\bar K_+=r$ имеем $\hat s=1$, $\hat p=\hat q$. При $n_+=0$: $\hat p=0$, а $s$ не оценивается.
\end{proposition}

\begin{proof}
По лемме~\ref{lem:split} максимизация распадается. $\ell_1$ --- биномиальное правдоподобие, максимум в $\hat q=n_+/N$. Для $\ell_2$: $\partial\ell_2/\partial s=0$ равносильно $\E(K\mid K\ge1)=\bar K_+$, то есть уравнению \eqref{eq:mle}; функция $s\mapsto rs/(1-(1-s)^r)$ непрерывна, строго возрастает и пробегает $(1,r)$ при $s\in(0,1)$ (это среднее усечённого бинома, монотонное по $s$ как среднее экспоненциального семейства), откуда единственность. Возвращаясь к $p$, получаем $\hat p=\hat q/(1-(1-\hat s)^r)$; если это число больше единицы, максимум по $p$ на $[0,1]$ достигается на границе $p=1$.
\end{proof}

\begin{remark}
Из \eqref{eq:mle} следует $\hat p\hat s=S/(rN)$: ОМП точно воспроизводит выборочное среднее $\bar K=S/N$, потому что $\E_\theta K=rps$. Второе условие --- совпадение доли релизов с красным: $\hat p\,(1-(1-\hat s)^r)=n_+/N$. Эти два равенства --- те же уравнения, что у метода моментов по статистикам $\Ind\{K\ge1\}$ и $K$.
\end{remark}

\section{EM-алгоритм}

Полные данные --- пары $(I_i,K_i)$. Их логарифмическое правдоподобие
\begin{equation}\label{eq:complete}
\ell_{\rm c}(\theta)=\sum_{i=1}^N\Bigl[I_i\bigl(\log p+K_i\log s+(r-K_i)\log(1-s)\bigr)+(1-I_i)\log(1-p)\Bigr]+\mathrm{const}
\end{equation}
линейно по $I_i$, поэтому E-шаг сводится к подстановке условных математических ожиданий $w_i=\E_{\theta}(I_i\mid K_i)=\Prob_\theta(I_i=1\mid K_i)$.

\paragraph{E-шаг.} При $K_i\ge1$ проблема есть наверняка: $w_i=1$. При $K_i=0$ по формуле Байеса
\begin{equation}\label{eq:estep}
w_i=w_0(\theta)=\frac{p(1-s)^r}{1-p+p(1-s)^r}.
\end{equation}

\paragraph{M-шаг.} Максимизируем $Q(\theta\mid\theta^{(t)})=\E_{\theta^{(t)}}\bigl(\ell_{\rm c}(\theta)\mid X\bigr)$, то есть \eqref{eq:complete} с $I_i$, заменёнными на $w_i$. Обозначим $W=\sum_iw_i=n_+ + n_0w_0(\theta^{(t)})$ --- «эффективное» число проблемных релизов. Тогда
\begin{equation}\label{eq:mstep}
p^{(t+1)}=\frac WN,\qquad s^{(t+1)}=\frac{\sum_iw_iK_i}{rW}=\frac S{rW}.
\end{equation}
Первое --- средний вес, второе --- доля красных среди прогонов на релизах, взвешенных по $w_i$; в числителе $\sum_iw_iK_i=S$, так как $K_i=0$ там, где $w_i<1$.

\paragraph{Алгоритм.} Начальное приближение $p^{(0)}=n_+/N$, $s^{(0)}=S/(rn_+)$ (релизы без красного считаем чистыми); затем повторять \eqref{eq:estep}--\eqref{eq:mstep} до стабилизации.

\begin{theorem}
Последовательность $\theta^{(t)}$ не убывает по правдоподобию: $\ell(\theta^{(t+1)})\ge\ell(\theta^{(t)})$. Любая её предельная точка с $n_+\ge1$ и $1<\bar K_+<r$ есть ОМП \eqref{eq:mle}.
\end{theorem}

\begin{proof}
Монотонность --- общее свойство EM (Dempster, Laird, Rubin, 1977): $\ell(\theta)=Q(\theta\mid\theta^{(t)})-H(\theta\mid\theta^{(t)})$, где $H(\theta\mid\theta^{(t)})=\E_{\theta^{(t)}}\bigl(\log\Prob_\theta(I\mid X)\mid X\bigr)$, а $H(\theta\mid\theta^{(t)})\le H(\theta^{(t)}\mid\theta^{(t)})$ по неравенству Йенсена; M-шаг увеличивает $Q$, значит, и $\ell$.

Пусть $\theta=(p,s)$ --- неподвижная точка \eqref{eq:mstep}: $p=W/N$, $s=S/(rW)$, где $W=n_++n_0w_0(\theta)$. Перемножая, получаем $ps=S/(rN)$. Далее, $W-n_+=n_0w_0$ и $N-W=n_0(1-w_0)$, поэтому
\[
\frac{W-n_+}{N-W}=\frac{w_0}{1-w_0}=\frac{p(1-s)^r}{1-p}.
\]
Подставим $W=Np$: $N-W=N(1-p)$, и равенство даёт $Np-n_+=Np(1-s)^r$, то есть $p\bigl(1-(1-s)^r\bigr)=n_+/N$. Итак, неподвижная точка решает систему $ps=S/(rN)$, $p(1-(1-s)^r)=n_+/N$ --- в точности уравнения ОМП из замечания после утверждения~\ref{prop:mle}; при $1<\bar K_+<r$ их решение единственно, а по монотонности правдоподобие сходится к его максимуму. Если же $\bar K_+=1$, EM уходит к границе $s\to0$, $p\to1$, как и ОМП.
\end{proof}

\begin{remark}[зачем EM, если есть явная формула]
Для одного теста EM не нужен: ответ даёт утверждение~\ref{prop:mle}. EM полезен, когда параметры связаны между тестами --- например, общая чувствительность $s$ для группы тестов. Тогда E-шаг по каждому тесту тот же, а M-шаг объединяет: $p_j=W_j/N$, $s=\sum_jS_j\big/\bigl(r\sum_jW_j\bigr)$; явной формулы здесь уже нет, а EM остаётся монотонным.
\end{remark}

\section{Что получается на данных задачи}

В опубликованной истории ($N=70$, $M=40$ тестов) у типичного теста $n_+=%(med)d$ релиза с красным. У $%(nzero)d$ тестов $n_+=0$: для них $\hat p=0$, а $s$ не определено. Вырожденный случай $\bar K_+=1$ (все релизы с красным дали ровно по одному красному) на опубликованной истории не встретился, но в истории того же мира хотя бы один такой тест есть в $%(dhist)d\%%$ случаев, в среднем $%(dmean).1f$ теста на историю: для него ОМП даёт $\hat s=0$, $\hat p=1$ --- тест выглядит как «ломается всегда, но почти не виден», и план, построенный по таким оценкам, гоняет его по $5$ раз. EM и явная формула совпадают на всех тестах с $n_+\ge1$ (расхождение не больше $10^{-%(dig)d}$).

\begin{center}\small
\begin{tabular}{@{}cccccccccc@{}}
\toprule
тест & $n_+$ & $S$ & $\hat p$ & $\hat s$ & $\hat p_{\rm EM}$ & $\hat s_{\rm EM}$ & итераций & $p$ & $s$ \\
\midrule
%(tab)s
\bottomrule
\end{tabular}
\end{center}

Последние два столбца --- истинные параметры генератора (команды их не видят). Видно главное: при двух-трёх событиях на тест разделить $p$ и $s$ по отдельному тесту нельзя --- оценки уходят на границу или промахиваются в разы, хотя произведение $\hat p\hat s=S/(rN)$ оценивается честно. Лечится это только общей информацией по тестам: общая $s$ в группе (EM из замечания выше), априорное распределение на $s$ по всем сорока тестам (эмпирический Байес) или подстановка общей $\hat s$ тестам без повторных красных, как сделано в эталонном решении методом моментов.

\end{document}
"""
import math
dig = max(1, int(-math.log10(max(dif, 1e-12))))
open(f"{root}/EM_один_тест.tex", "w").write(tex % dict(med=int(np.median(nplus)), nzero=len(zero), dhist=round(100 * cnt_hist / 200), dmean=cnt_tests / 200, dig=dig, tab=tabrows))
print("tex written")
