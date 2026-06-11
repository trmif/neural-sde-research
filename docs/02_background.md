```{mermaid}
flowchart LR
    subgraph B1["Ветка 1 — Latent Variable Models"]
        A1["**D. Kingma**\nVAE\n2013"] --> A2["**S. Sohl-Dickstein**\nDiffusion\n2015"] --> A3["**J. Ho**\nDDPM\n2020"]
    end

    subgraph B2["Ветка 2 — Score-based Models"]
        C1["**P. Vincent**\nDenoising Score\nMatching\n2013"] --> C2["**Y. Song**\nAnnealed Langevin\nDynamics\n2019"] --> C3["**Y. Song**\nScore-based\nDiffusion\n2021"]
    end

    subgraph B3["Ветка 3 — Neural SDE"]
        D1["**D. Rezende**\nDLGM\n2014"] --> D2["**T. Chen**\nNeural ODE\n2018"] --> D3["**Tzen & Raginsky**\nNeural SDE\n2019"]
    end

    A3 --> E["**Современные\nгенеративные\nмодели**"]
    C3 --> E
    D3 --> E
```
