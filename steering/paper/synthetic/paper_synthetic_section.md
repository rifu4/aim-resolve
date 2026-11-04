# Synthetic data test in the paper: draft for main.tex

Status: 2026-11-02. Draft, nothing in `main.tex` has been changed.

**Values:** the full-resolution test with real priors has not been run. Everything inside `\XX{...}` is a current low-resolution value or an open choice, and must be replaced. To render them while drafting, add this to the preamble:

```latex
\usepackage{xcolor}
\newcommand{\XX}[1]{\textcolor{red}{#1}}
```


## 1. Placement

The test comes before the real results, so the reader sees the method validated first and can judge the real maps with its limits in mind. For that, the observation and model setup have to come before the test, so `sec:imaging` is split in two:

| new structure of `\section{Applications}` | content |
|---|---|
| `\subsection{MeerKAT observation and model setup} \label{sec:setup}` | The opening paragraphs of `sec:imaging` and `M\&M prior model and inference`, with `tab:components` and `tab:frequencies`, unchanged apart from Part 4 |
| `\subsection{Validation on synthetic data} \label{sec:synthetic}` | New, Part 2 |
| `\subsection{Spatio-spectral imaging of MeerKAT data} \label{sec:imaging}` | `Multi-frequency imaging results` and `Comparison of the spectral index maps` |
| `\subsection{Spectral analysis} \label{sec:analysis}` | Unchanged apart from the caveats in Part 3 |

The truth is built from the real reconstruction, which the test refers to as a forward reference to `sec:imaging`. The test uses no real-data results otherwise.

| where | change |
|---|---|
| `sec:introduction`, last paragraph | One sentence for the new subsection (Part 3) |
| `sec:mf-prior` | **Rewrite needed:** the model now has an explicit curvature field (Part 4) |
| `sec:likelihood` | Add the primary beam, the weight correction and the noise model (Part 4) |
| `sec:mf-fast-inference` | Add the per-channel residual (Part 4) |
| `sec:mf-transition` | Add the noise reset (Part 4) |
| `sec:applications`, first paragraph | Add the roadmap sentence (Part 3) |
| `sec:setup` | Update to 8 model frequencies, 1029.7 MHz reference, 468 channels (Part 4) |
| `sec:imaging` opening | Split into `sec:setup` and `sec:imaging` (see above) |
| New `sec:synthetic` | Full text below (Part 2) |
| `sec:curvature`, `sec:c2-analysis` | Add the caveat at the edge of the field (Part 3) |
| `\section{Conclusions}` | Add a paragraph (Part 3) |


## 2. New subsection

Insert between the new `sec:setup` and `sec:imaging`.

```latex
\subsection{Validation on synthetic data} \label{sec:synthetic}

Before we apply the method to the MeerKAT observation, we test whether it recovers known spectral maps from data with the same properties. The test addresses three questions: how accurately are the spectral index and the spectral curvature recovered, does this accuracy depend on the position in the primary beam, and are the posterior uncertainties reliable.


\subsubsection{Synthetic data} \label{sec:synthetic-data}

We built the synthetic sky from the $1\degr \times 1\degr$ field around the phase center, which contains ESO137-006 and ESO137-007. We used the brightness of the two galaxies and of the background from our reconstruction in \cref{sec:imaging} as the ground truth at the reference frequency, so that the test has the morphology and the dynamic range of the real sky. Each of the three components received its own spectrum,
\begin{equation} \label{eq:synthetic-truth}
   I(x, \nu) = \sum_{k} I^{k}(x, \bar{\nu}) \exp\left[ \alpha^{k}(x)\, u + \beta^{k}(x)\, u^2 \right], \qquad u = \ln \frac{\nu}{\bar{\nu}},
\end{equation}
where $k$ runs over the background and the two galaxies. The fields $\alpha^{k}$ and $\beta^{k}$ are independent Gaussian random fields with a correlation length of $45\arcsec$, a mean of $-1.0$ and $-0.2$, and a standard deviation of $0.5$ and $0.3$, respectively. They do not follow the prior of the reconstruction, and the truth does not contain a deviation term.

For each of the 468 data channels, we multiplied the sky at the exact channel frequency by the primary beam of that channel and applied the response function with the real $uvw$ coordinates, weights, and flags. To keep the computation tractable, we used \XX{10\%} of the visibility rows, with the weights scaled such that the signal-to-noise ratio per model frequency is the same as in the full data. We then added Gaussian noise with a standard deviation of $s_c / \sqrt{w}$, where $w$ are the visibility weights and $s_c$ is the noise level measured per channel on the longest baselines of the real data (see \cref{sec:likelihood}). The synthetic data are therefore not generated from the binned model of the inference: the spectra vary within each model frequency bin, exactly as for real data.


\subsubsection{Reconstruction} \label{sec:synthetic-reconstruction}

We reconstructed the synthetic data with the setup of \cref{sec:setup}: a background field and one extended component for each galaxy, the same prior parameters, the transition from a single-frequency stage to the multi-frequency model, eight model frequencies, and the same number of major iterations. As the synthetic field has a single bright pair of sources, we did not run the source detection of \texttt{aim-resolve} and placed the components at the positions of the real ones. \XX{The starting point of the single-frequency stage was a fit to the \{apparent / intrinsic\} truth.}


\subsubsection{Recovered spectra} \label{sec:synthetic-results}

\Cref{fig:synthetic-maps} compares the recovered maps of the spectral index and the spectral curvature to the ground truth, and \cref{tab:synthetic} lists the errors for the two galaxies and as a function of the distance from the phase center.

For ESO137-006, located near the phase center, both quantities are recovered with high accuracy. In the brightest quartile of the pixels, the median error is \XX{$+0.001$} for the spectral index and \XX{$-0.002$} for the curvature, with a typical absolute error of \XX{$0.004$} and \XX{$0.02$}. For ESO137-007, the errors grow with the distance from the phase center. Inside a radius of \XX{$0.35\degr$}, the median curvature error stays below \XX{$0.02$}, while at larger radii, where the primary beam drops below about \XX{0.8} and the source is faint, the result depends on the spectrum of the surrounding emission (see below): the median error is \XX{$+0.02$} for a truth with individual component spectra and reaches \XX{$-0.2$} if all components share one spectrum. We take this range as a systematic uncertainty of the curvature at the outer end of ESO137-007.

The fit to the data is close to the noise level. The exact reduced $\chi^2$, computed per data channel with the response function, is \XX{$1.02$} overall, and \XX{$1.07$} on the shortest baselines (below $250\lambda$), compared to $1.000$ for the true sky.


\subsubsection{Component separation} \label{sec:synthetic-separation}

The total flux of the field is recovered to \XX{1\%}. The split between components is less well constrained for faint diffuse emission that overlaps with a source. At the outer end of ESO137-007, the background holds about twice its true share of the flux, \XX{$0.26$} instead of $0.12$, while the galaxy component is correspondingly low, and the summed flux in this region is \XX{5--8\%} too high. Over the whole field the background is \XX{16--23\%} too bright. This excess is not supported by the data, as the true sky fits the short baselines better than the reconstruction. \XX{[Cause, once known: noise / prior / start of the inference.]} The bright parts of the galaxies are not affected.


\subsubsection{Calibration of the uncertainties} \label{sec:synthetic-uncertainties}

For Gaussian posteriors, $68\%$ of the pixels should lie within one standard deviation of the truth. Averaged over pixels above a brightness of $0.01\,\text{mJy\,arcsec}^{-2}$, we find \XX{$58\%$} for the spectral index and \XX{$44$--$55\%$} for the curvature, and the error distribution has broader tails than a Gaussian. Our posterior standard deviations therefore underestimate the true errors, by a factor of about \XX{1.5--2} at the $68\%$ level. Part of this is expected, since variational inference tends to underestimate posterior widths and we approximate the posterior with a small number of samples. We applied this factor when we interpret differences in \cref{sec:analysis}. \XX{[Check that this is what we do.]}


\subsubsection{Test without curvature} \label{sec:synthetic-beta0}

To test whether the model invents curvature, we repeated the reconstruction with a truth with $\beta^{k} = 0$ everywhere. \XX{[Result: recovered curvature as a function of radius.]}
```

### Table

```latex
\begin{table}[h]
\caption{Errors of the recovered spectra on the synthetic data, in regions of the distance $r$ from the phase center.}
\label{tab:synthetic}
\centering
\begin{tabular}{l c c c c}
\hline\hline
\noalign{\vskip 4pt}
Region & $\Delta\alpha$ & $\Delta\beta$ & $\sigma_\beta$ & Flux ratio \\
\noalign{\vskip 4pt}
\hline
\noalign{\vskip 4pt}
   ESO137-006 & \XX{$+0.00$} & \XX{$-0.00$} & \XX{$0.05$} & \XX{$0.99$} \\
   ESO137-007 $r < 0.25\degr$ & \XX{} & \XX{} & \XX{} & \XX{$1.00$} \\
   ESO137-007 $0.30\degr < r < 0.35\degr$ & \XX{} & \XX{$-0.02$} & \XX{$0.22$} & \XX{$1.05$} \\
   ESO137-007 $r > 0.35\degr$ & \XX{} & \XX{$+0.02$} & \XX{$0.22$} & \XX{$1.07$} \\
\noalign{\vskip 2pt}
\hline
\end{tabular}
\tablefoot{$\Delta$ denotes the median of reconstruction minus truth, $\sigma_\beta$ the median posterior standard deviation. The flux ratio is the summed flux of all components relative to the truth at the reference frequency.}
\end{table}
```

### Figure

One two-column figure: rows ESO137-006 and ESO137-007, columns truth, reconstruction, difference for alpha and beta, plus a lower panel with beta error against radius. `compare_synthetic.py`, `compare_component.py` and `check_beta_radius.py` already produce the content.

```latex
\begin{figure*}
\centering
   \includegraphics[width=17cm]{synthetic_maps.png}
   \caption{Reconstruction of the synthetic data. For ESO137-006 and ESO137-007, the panels show the true spectral index and spectral curvature, the reconstructions, and their differences. The lower panel shows the curvature error as a function of the distance from the phase center, with the posterior standard deviation as the shaded region.}
   \label{fig:synthetic-maps}
\end{figure*}
```


## 3. Changes in the existing text

### Introduction, last paragraph

Replace the last two sentences with:

```latex
In \cref{sec:applications}, we first show the application of our method to the MeerKAT observation mentioned before. We first describe this observation and the model setup, and validate the recovered spectra on synthetic data with the same properties. We then present the imaging results of our method and compare our recovered spectral index maps to the post-processed maps of \citet{dabbech2022}, which are available as \texttt{FITS} files. Finally, we perform a spectral analysis of the two galaxies ESO137-006 and ESO137-007 using the posterior samples of our reconstruction, including their spectral index and spectral curvature.
```

### Applications, first paragraph

Replace the sentences after the first one with:

```latex
In \cref{sec:setup}, we describe the observation and the model setup. In \cref{sec:synthetic}, we validate the recovered spectra and their uncertainties on synthetic data with the same properties. In \cref{sec:imaging}, we present the reconstructed sky brightness distributions of the individual sky components over several frequency channels and compare our spectral index maps to the post-processed maps of \citet{dabbech2022}. In \cref{sec:analysis}, we infer spectral curvature maps from our spectral model and use them, together with the spectral index maps and their uncertainty estimates, to perform a spectral analysis of the galaxies ESO137-006 and ESO137-007.
```

### Spectral curvature, end of the paragraph on `fig:cs-curvature`

```latex
As shown in \cref{sec:synthetic}, the accuracy of these maps depends on the position in the field.
```

### Analysis of ESO137-007, after the paragraph on `fig:c2_curvature_alpha`

```latex
These statements have to be read together with the validation in \cref{sec:synthetic}. The outer half of the tail lies at a distance of more than \XX{$0.35\degr$} from the phase center, where the curvature of faint emission carries a systematic uncertainty of about \XX{$0.1$--$0.2$} on the synthetic data. \XX{[Only if the rerun confirms it:] The change of sign from positive to negative curvature along the tail is larger than this and is robust, whereas the exact level of $\beta \simeq -0.3$ to $-0.4$ and the flattening at the end of the tail are not constrained at this accuracy.}
```

### Conclusions, after the paragraph on the spectral analysis

```latex
We validated the method on synthetic data with the real $uv$ coverage, the real noise level, and individual spectra for each component. The spectral index and the spectral curvature of bright emission are recovered with high accuracy. Toward the edge of the field, the curvature of faint emission carries a systematic uncertainty of about \XX{$0.1$--$0.2$}, and faint diffuse flux can be assigned to the background instead of a source. Moreover, the posterior standard deviations underestimate the true errors by a factor of about \XX{1.5--2}.
```


## 4. Methods updates required by the new setup

`main.tex` still describes the earlier setup. These are needed whether or not the test goes in, because the test refers to the setup of `sec:setup`.

### `sec:mf-prior`: curvature model

Replace `eq:mf-model` and its explanation. Current code (`SpectralIndexCurvature`, `mf_ext_8f.yml`):

```latex
\begin{equation} \label{eq:mf-model}
   \ln I^{\text{b}}(x, \nu) = \ln I^{\text{b}}(x, \bar{\nu}) + \alpha^{\text{b}}(x)\, u + \beta^{\text{b}}(x)\, q(u) + \delta^{\text{b}}(x, \nu),
\end{equation}
where $u = \ln(\nu/\bar{\nu})$ and $q(u) = u^2 - c\,u$ is $u^2$ orthogonalized against $u$ over the model frequencies, with $c = \sum u^3 / \sum u^2$. Here, $I^{\text{b}}(x, \bar{\nu})$ describes the sky brightness at the reference frequency $\bar{\nu}$, $\alpha^{\text{b}}(x)$ is the spectral index map, $\beta^{\text{b}}(x)$ the spectral curvature map, and $\delta^{\text{b}}(x, \nu)$ captures the remaining deviations. Since $q$ is orthogonal to $u$, $\alpha$ is the slope of the spectrum and $\beta$ the coefficient of $u^2$. The deviations are projected to be orthogonal to both $u$ and $q$, so they carry neither slope nor curvature. $I^{\text{b}}(x, \bar{\nu})$, $\alpha^{\text{b}}(x)$, and $\beta^{\text{b}}(x)$ are modeled with two-dimensional GPs, where $\alpha^{\text{b}}$ and $\beta^{\text{b}}$ share one spatial power spectrum with separate amplitudes, while $\delta^{\text{b}}(x, \nu)$ is described by a Wiener process in the frequency direction for each spatial location $x$.
```

Consequences:

- `sec:curvature` says the curvature is inferred by fitting Eq. (curvature) to the samples, because the deviation term carries it. With the new model, beta is a direct model parameter, so those two paragraphs need rewriting.
- `sec:comparison` says each component's spectral index has "its own correlation structure". That still holds between components, but alpha and beta of one component now share it.
- `app:params` needs the curvature priors: mean (0, 0.5), fluctuations (1.0, 0.5), and beta (0, 0.5) for point sources.

### `sec:likelihood`: beam, weights, noise model

After `eq:rime`:

```latex
Here we omitted the primary beam of the antennas for readability. In our reconstructions, $I$ denotes the intrinsic sky, and the response includes the frequency-dependent MeerKAT primary beam, $I(l,m,\nu) \to A(l,m,\nu)\, I(l,m,\nu)$, with the Stokes I beam computed once for each model frequency.
```

After `eq:ham`:

```latex
The weights of radio interferometric data are often not an accurate description of the actual noise. We therefore measured the noise level $s_c$ of each data channel as the robust scatter of the data on the longest baselines, relative to $1/\sqrt{w}$, and rescaled the weights accordingly, $w \to w / s_c^2$. For our data, $s_c$ varies between about 0.5 and 1.1 across the band.
```

After the equation for `N'^{-1}`:

```latex
To absorb remaining errors of the noise description, we additionally multiply the inverse noise kernel by a learned scale for each Fourier mode, with an inverse-gamma prior with a mode of \XX{1.0} and a mean of \XX{1.1}.
```

### `sec:mf-fast-inference`: per-channel residual

Replace "Apart from that, the \texttt{fast-resolve} inference scheme remains unchanged." with:

```latex
A single image per model frequency cannot describe the change of the sky across the data channels of one bin. In the major cycles, we therefore compute the exact residual channel by channel: the sky of the corresponding model frequency is extrapolated to each data channel with its local spectral slope, multiplied by the primary beam of that channel, and passed through the response function. The minor cycles remain on the $c_m$ model frequencies. On synthetic data, this reduces the reduced $\chi^2$ of the true sky from 1.68 to 1.00 and leaves no measurable error on any baseline range. Apart from that, the \texttt{fast-resolve} inference scheme remains unchanged.
```

### `sec:mf-transition`: noise reset

Add at the end of the second paragraph:

```latex
The learned noise scale of the single-frequency stage is not transferred. It restarts from its prior, since the single-frequency model cannot describe the spectrum of the data and part of this misfit ends up in the noise scale, in particular on the short baselines.
```

### `sec:setup`

- Eight model frequencies (four per band) instead of six, reference 1029.7 MHz instead of 1054 MHz.
- Update `tab:frequencies`, the "six model frequencies" sentence of `fig:sky-rgb`, and the 1054 MHz mentions in the figures and appendix.
- Update the iteration and sample numbers once the real run is final.
- All real-data numbers in `sec:analysis` come from the earlier run and have to be re-derived.


## 5. Points to decide before submission

1. **The curvature claims in the abstract and conclusions.** "Clear transition from positive to negative curvature along the tail" should survive, but check it against the rerun. The bias at the tail end has the same sign as the negative curvature claimed there, so the outer values need the caveat in Part 3.
2. **"Cleanly separates the different components" in the abstract.** The test shows a real limit for faint diffuse flux at the edge. Softening it to something like "separates the different components" avoids a contradiction with the new subsection.
3. **`sec:synthetic-reconstruction`** assumes components at the known positions. If you want to run the full detection on synthetic data too, that is a different and longer test.
4. **The range for the systematic.** The text uses the spread between the shared and the per-component truth. That needs the second noise seed to be defensible.
5. **The real-data check.** A sentence on the self-template fit could go in `sec:synthetic-uncertainties` once it has been run on the real result.
