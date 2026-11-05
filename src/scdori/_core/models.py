import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class scDoRI(nn.Module):
    """
    The scDORI model integrates single cell multi-ome RNA and ATAC data to learn latent topic representations and perform gene regulatory network (GRN) inference.

    This model contains:
    - **Encoders** for RNA and ATAC, producing a shared topic distribution.
    - **Decoders** for ATAC, TF, and RNA reconstruction.
    - **GRN logic** for combining TF binding data with gene-peak links and tf expression to reconstruct RNA profiles.

    Parameters
    ----------
    device : torch.device
        The device (CPU or CUDA) for PyTorch operations.
    num_genes : int
        Number of genes in the RNA data.
    num_peaks : int
        Number of peaks in the ATAC data.
    num_tfs : int
        Number of transcription factors being modeled.
    num_topics : int
        Number of latent topics or factors.
    num_batches : int
        Number of distinct batches (for batch correction).
    dim_encoder1 : int
        Dimension of the first encoder layer.
    dim_encoder2 : int
        Dimension of the second encoder layer.
    batch_norm : bool, optional
        If True, use batch normalization in encoder and library factor MLPs. Default is True.

    Attributes
    ----------
    encoder_rna : torch.nn.Sequential
        The neural network layers for the RNA encoder.
    encoder_atac : torch.nn.Sequential
        The neural network layers for the ATAC encoder.
    mu_theta : torch.nn.Linear
        Linear layer converting combined RNA+ATAC encoder outputs into raw topic logits.
    topic_peak_decoder : torch.nn.Parameter
        A (num_topics x num_peaks) parameter for ATAC reconstruction.
    atac_batch_factor : torch.nn.Parameter
        A (num_batches x num_peaks) parameter for batch effects in ATAC.
    atac_batch_norm : torch.nn.BatchNorm1d
        Batch normalization layer for ATAC predictions.
    topic_tf_decoder : torch.nn.Parameter
        A (num_topics x num_tfs) parameter for TF expression reconstruction.
    tf_batch_factor : torch.nn.Parameter
        A (num_batches x num_tfs) parameter for batch effects in TF reconstruction.
    tf_batch_norm : torch.nn.BatchNorm1d
        Batch normalization layer for TF predictions.
    tf_alpha_nb : torch.nn.Parameter
        A (1 x num_tfs) parameter for TF negative binomial overdispersion.
    gene_peak_factor_learnt : torch.nn.Parameter
        A (num_genes x num_peaks) learned matrix linking peaks to genes.
    gene_peak_factor_fixed : torch.nn.Parameter
        A (num_genes x num_peaks) fixed mask for feasible gene-peak links.
    rna_batch_factor : torch.nn.Parameter
        A (num_batches x num_genes) parameter for batch effects in RNA reconstruction.
    rna_batch_norm : torch.nn.BatchNorm1d
        Batch normalization layer for RNA predictions.
    rna_alpha_nb : torch.nn.Parameter
        A (1 x num_genes) parameter for RNA negative binomial overdispersion.
    tf_library_factor : torch.nn.Sequential
        An MLP to predict library scaling factor for TF data from the observed TF expression.
    rna_library_factor : torch.nn.Sequential
        An MLP to predict library scaling factor for RNA data from the observed gene counts.
    tf_binding_matrix_activator : torch.nn.Parameter
        A (num_peaks x num_tfs) matrix of in silico ChIP-seq (activator) TF-peak binding; precomputed and fixed.
    tf_binding_matrix_repressor : torch.nn.Parameter
        A (num_peaks x num_tfs) matrix of in silico ChIP-seq (repressor) TF-peak binding; precomputed and fixed.
    tf_gene_topic_activator_grn : torch.nn.Parameter
        A (num_topics x num_tfs x num_genes) matrix capturing per-topic activator regulation.
    tf_gene_topic_repressor_grn : torch.nn.Parameter
        A (num_topics x num_tfs x num_genes) matrix capturing per-topic repressor regulation.
    rna_grn_batch_factor : torch.nn.Parameter
        A (num_batches x num_genes) batch-effect parameter for the GRN-based RNA reconstruction (module 4).
    rna_grn_batch_norm : torch.nn.BatchNorm1d
        Batch normalization layer for GRN-based RNA predictions.
    """

    def __init__(
        self,
        device,
        num_genes,
        num_peaks,
        num_tfs,
        num_topics,
        num_batches,
        dim_encoder1,
        dim_encoder2,
        batch_norm=True,
    ):
        super().__init__()
        self.device = device
        self.num_genes = num_genes
        self.num_peaks = num_peaks
        self.num_tfs = num_tfs
        self.num_topics = num_topics
        # Optional GRN-phase topic subset (original IDs); None uses all topics.
        self.grn_topics = None
        self.num_batches = num_batches
        self.dim_encoder1 = dim_encoder1
        self.dim_encoder2 = dim_encoder2
        self.batch_norm = batch_norm

        # ENCODER for RNA
        self.encoder_rna = nn.Sequential(
            nn.Linear(num_genes + 2, dim_encoder1),
            nn.BatchNorm1d(dim_encoder1) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(dim_encoder1, dim_encoder2),
            nn.BatchNorm1d(dim_encoder2) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        # ENCODER for ATAC
        self.encoder_atac = nn.Sequential(
            nn.Linear(num_peaks + 2, dim_encoder1),
            nn.BatchNorm1d(dim_encoder1) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(dim_encoder1, dim_encoder2),
            nn.BatchNorm1d(dim_encoder2) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.mu_theta = nn.Linear(dim_encoder2 * 2, num_topics)

        # ATAC decoder (module 1)
        self.topic_peak_decoder = nn.Parameter(torch.rand(num_topics, num_peaks))
        self.atac_batch_factor = nn.Parameter(torch.rand(num_batches, num_peaks))
        self.atac_batch_norm = nn.BatchNorm1d(num_peaks)

        # RNA from ATAC (module 2)
        self.gene_peak_factor_learnt = nn.Parameter(torch.rand(num_genes, num_peaks))
        self.gene_peak_factor_fixed = nn.Parameter(torch.ones(num_genes, num_peaks))
        self.rna_batch_factor = nn.Parameter(torch.rand(num_batches, num_genes))
        self.rna_batch_norm = nn.BatchNorm1d(num_genes)
        self.rna_alpha_nb = nn.Parameter(torch.rand(1, num_genes))

        # TF decoder (module 3)
        self.topic_tf_decoder = nn.Parameter(torch.rand(num_topics, num_tfs))
        self.tf_batch_factor = nn.Parameter(torch.rand(num_batches, num_tfs))
        self.tf_batch_norm = nn.BatchNorm1d(num_tfs)
        self.tf_alpha_nb = nn.Parameter(torch.rand(1, num_tfs))

        # MLP for library factor (TF, RNA)
        self.tf_library_factor = nn.Sequential(
            nn.Linear(num_tfs, dim_encoder2),
            nn.BatchNorm1d(dim_encoder2) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(dim_encoder2, dim_encoder1),
            nn.BatchNorm1d(dim_encoder1) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(dim_encoder1, 1),
            nn.Softplus(),
        )
        self.rna_library_factor = nn.Sequential(
            nn.Linear(num_genes, dim_encoder2),
            nn.BatchNorm1d(dim_encoder2) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(dim_encoder2, dim_encoder1),
            nn.BatchNorm1d(dim_encoder1) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(dim_encoder1, 1),
            nn.Softplus(),
        )

        # GRN portion (module 4)
        self.tf_binding_matrix_activator = nn.Parameter(torch.rand(num_peaks, num_tfs))
        self.tf_binding_matrix_repressor = nn.Parameter(torch.rand(num_peaks, num_tfs))
        self.tf_gene_topic_activator_grn = nn.Parameter(torch.rand(num_topics, num_tfs, num_genes))
        self.tf_gene_topic_repressor_grn = nn.Parameter(torch.rand(num_topics, num_tfs, num_genes))
        self.rna_grn_batch_factor = nn.Parameter(torch.rand(num_batches, num_genes))
        self.rna_grn_batch_norm = nn.BatchNorm1d(num_genes)

    def encode(self, rna_input, atac_input, log_lib_rna, log_lib_atac, num_cells):
        """
        Encode RNA and ATAC input into a topic distribution (theta).

        Parameters
        ----------
        rna_input : torch.Tensor
            A (B, num_genes) tensor of RNA counts per cell.
        atac_input : torch.Tensor
            A (B, num_peaks) tensor of ATAC counts per cell.
        log_lib_rna : torch.Tensor
            A (B, 1) tensor of log RNA library sizes.
        log_lib_atac : torch.Tensor
            A (B, 1) tensor of log ATAC library sizes.
        num_cells : torch.Tensor
            A (B, 1) tensor representing how many cells are aggregated (if metacells),
            or all ones for single-cell data.

        Returns
        -------
        (theta, mu_theta) : tuple of torch.Tensor
            theta    : (B, num_topics), softmaxed topic distribution.
            mu_theta : (B, num_topics), raw topic logits.
        """
        B = rna_input.shape[0]
        # Concat RNA input, log_lib_rna, and num_cells
        x_rna = torch.cat([rna_input, log_lib_rna.view(B, 1), num_cells.view(B, 1)], dim=1)
        x_atac = torch.cat([atac_input, log_lib_atac.view(B, 1), num_cells.view(B, 1)], dim=1)
        qrna = self.encoder_rna(x_rna)
        qatac = self.encoder_atac(x_atac)
        combined = torch.cat([qrna, qatac], dim=1)
        mu_theta = self.mu_theta(combined)
        theta = F.softmax(mu_theta, dim=-1)
        return theta, mu_theta

    def topic_gene_matrix(self, gene_peak=None):
        """Compute the topic-to-gene decoder, retaining the reference reduction order."""
        peaks = F.softmax(self.topic_peak_decoder, dim=1)
        minimum, _ = torch.min(peaks, dim=0, keepdim=True)
        maximum, _ = torch.max(peaks, dim=0, keepdim=True)
        scaled = (peaks - minimum) / (maximum - minimum + 1e-8)
        if gene_peak is None:
            gene_peak = (self.gene_peak_factor_learnt * self.gene_peak_factor_fixed).T
        return torch.mm(scaled, gene_peak)

    def forward(
        self,
        rna_input,
        atac_input,
        tf_input,
        topic_tf_input,
        log_lib_rna,
        log_lib_atac,
        num_cells,
        batch_onehot,
        phase="warmup_1",
        *,
        fixed_topic_gene=None,
    ):
        """
        Forward pass through scDORI, producing predictions for ATAC, TF, and RNA reconstructions (Phase 1), as well as GRN-based RNA predictions in GRN phase (Phase 2).

        Parameters
        ----------
        rna_input : torch.Tensor
            Shape (B, num_genes). RNA counts per cell in the batch.
        atac_input : torch.Tensor
            Shape (B, num_peaks). ATAC counts per cell in the batch.
        tf_input : torch.Tensor
            Shape (B, num_tfs). Observed TF expression.
        topic_tf_input : torch.Tensor
            Shape (num_topics, num_tfs). TF expression aggregated by topic,
            used only if phase == "grn".
        log_lib_rna : torch.Tensor
            Shape (B, 1). Log of RNA library sizes.
        log_lib_atac : torch.Tensor
            Shape (B, 1). Log of ATAC library sizes.
        num_cells : torch.Tensor
            Shape (B, 1). Number of cells aggregated (if metacells), else ones.
        batch_onehot : torch.Tensor
            Shape (B, num_batches). One-hot batch encoding for each cell.
        phase : str, optional
            Which training phase: "warmup_1", "warmup_2", or "grn".
            If phase=="grn", the GRN-based RNA predictions are included.
        fixed_topic_gene : torch.Tensor, optional
            Precomputed topic-to-gene decoder for frozen, no-gradient evaluation.
            The caller must rebuild it after changing phase 1 weights.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - "theta": (B, num_topics), the softmaxed topic distribution.
            - "mu_theta": (B, num_topics), raw topic logits.
            - "preds_atac": (B, num_peaks), predicted peak accessibility.
            - "preds_tf": (B, num_tfs), predicted TF expression.
            - "mu_nb_tf": (B, num_tfs), TF negative binomial mean = preds_tf * TF library factor.
            - "preds_rna": (B, num_genes), predicted RNA expression.
            - "mu_nb_rna": (B, num_genes), RNA negative binomial mean = preds_rna * RNA library factor.
            - "preds_rna_from_grn": (B, num_genes), optional GRN-based RNA predictions.
            - "mu_nb_rna_grn": (B, num_genes), negative binomial mean of GRN-based RNA predictions.
            - "library_factor_tf": (B, 1), predicted library factor for TF.
            - "library_factor_rna": (B, 1), predicted library factor for RNA.

        """
        # 1) ENCODE => topic distribution
        theta, mu_theta = self.encode(rna_input, atac_input, log_lib_rna, log_lib_atac, num_cells)

        # 2) ATAC decoding
        batch_factor_atac = torch.mm(batch_onehot, self.atac_batch_factor)
        preds_atac = torch.mm(theta, self.topic_peak_decoder) + batch_factor_atac
        preds_atac = self.atac_batch_norm(preds_atac)
        preds_atac = F.softmax(preds_atac, dim=-1)

        # 3) TF decoding => library factor
        batch_factor_tf = torch.mm(batch_onehot, self.tf_batch_factor)
        tf_logits = torch.mm(theta, self.topic_tf_decoder) + batch_factor_tf
        tf_logits = self.tf_batch_norm(tf_logits)
        preds_tf = F.softmax(tf_logits, dim=-1)
        # library MLP for TF
        library_factor_tf = self.tf_library_factor(tf_input)
        mu_nb_tf = preds_tf * library_factor_tf

        # 4) RNA from ATAC => library factor
        gene_peak = (self.gene_peak_factor_learnt * self.gene_peak_factor_fixed).T if phase == "grn" else None
        batch_factor_rna = torch.mm(batch_onehot, self.rna_batch_factor)
        if fixed_topic_gene is not None:
            if (
                torch.is_grad_enabled()
                or self.training
                or any(
                    p.requires_grad
                    for p in (self.topic_peak_decoder, self.gene_peak_factor_learnt, self.gene_peak_factor_fixed)
                )
            ):
                raise ValueError("fixed_topic_gene requires frozen decoder parameters and no-gradient evaluation")
            topicxgene = fixed_topic_gene
        else:
            topicxgene = self.topic_gene_matrix(gene_peak)
        rna_logits = torch.mm(theta, topicxgene) + batch_factor_rna
        rna_logits = self.rna_batch_norm(rna_logits)
        preds_rna = F.softmax(rna_logits, dim=-1)

        # library MLP for RNA
        library_factor_rna = self.rna_library_factor(rna_input)
        mu_nb_rna = preds_rna * library_factor_rna

        # 5) GRN => preds_rna_from_grn if phase=="grn"
        if phase == "grn":
            topic_peak_denoised1 = F.softmax(self.topic_peak_decoder, dim=1)
            topics = range(self.num_topics) if self.grn_topics is None else self.grn_topics.tolist()
            grn_atac_activator = torch.empty(size=(len(topics), self.num_tfs, self.num_genes)).to(self.device)
            grn_atac_repressor = torch.empty(size=(len(topics), self.num_tfs, self.num_genes)).to(self.device)

            # Calculate ATAC-based TF–gene links (activator/repressor) for each (selected) topic
            for i, topic in enumerate(topics):
                topic_gene_peak = topic_peak_denoised1[topic][:, None] * gene_peak
                G_topic = self.tf_binding_matrix_activator.T @ topic_gene_peak
                G_topic = G_topic / (gene_peak.sum(axis=0, keepdims=True) + 1e-7)
                grn_atac_activator[i] = G_topic

                topic_gene_peak = (1 / (topic_peak_denoised1[topic] + 1e-20))[:, None] * gene_peak
                G_topic = self.tf_binding_matrix_repressor.T @ topic_gene_peak
                G_topic = G_topic / (gene_peak.sum(axis=0, keepdims=True) + 1e-7)
                grn_atac_repressor[i] = G_topic

            C = torch.empty(size=(len(topics), self.num_genes)).to(self.device)
            tf_expression_input = topic_tf_input.to(self.tf_gene_topic_activator_grn)
            for i, topic in enumerate(topics):
                gene_atac_activator_topic = grn_atac_activator[i] / (grn_atac_activator[i].max() + 1e-15)
                gene_atac_repressor_topic = grn_atac_repressor[i] / (grn_atac_repressor[i].min() + 1e-15)

                G_act = gene_atac_activator_topic * torch.nn.functional.relu(self.tf_gene_topic_activator_grn[topic])
                G_rep = (
                    gene_atac_repressor_topic * -1 * torch.nn.functional.relu(self.tf_gene_topic_repressor_grn[topic])
                )

                C[i] = tf_expression_input[topic] @ G_act + tf_expression_input[topic] @ G_rep

            batch_factor_rna_grn = torch.mm(batch_onehot, self.rna_grn_batch_factor)
            # Excluded topics drop theta[:, excluded] @ C[excluded]; see scdori._core.topic_selection.
            preds_rna_from_grn = torch.mm(theta if self.grn_topics is None else theta[:, topics], C)
            preds_rna_from_grn = preds_rna_from_grn + batch_factor_rna_grn
            preds_rna_from_grn = self.rna_grn_batch_norm(preds_rna_from_grn)
            preds_rna_from_grn = nn.Softmax(dim=1)(preds_rna_from_grn)
        else:
            preds_rna_from_grn = torch.zeros_like(preds_rna)

        mu_nb_rna_grn = preds_rna_from_grn * library_factor_rna

        return {
            "theta": theta,
            "mu_theta": mu_theta,
            "preds_atac": preds_atac,
            "preds_tf": preds_tf,
            "mu_nb_tf": mu_nb_tf,
            "preds_rna": preds_rna,
            "mu_nb_rna": mu_nb_rna,
            "preds_rna_from_grn": preds_rna_from_grn,
            "mu_nb_rna_grn": mu_nb_rna_grn,
            "library_factor_tf": library_factor_tf,
            "library_factor_rna": library_factor_rna,
        }


def initialize_scdori_parameters(
    model,
    gene_peak_distance_exp: torch.Tensor,
    gene_peak_fixed: torch.Tensor,
    insilico_act: torch.Tensor,
    insilico_rep: torch.Tensor,
    phase="warmup",
):
    """
    Initialize or freeze certain scDORI parameters, preparing for either warmup or GRN phases.

    Parameters
    ----------
    model : torch.nn.Module
        An instance of the scDORI model.
    gene_peak_distance_exp : torch.Tensor
        Shape (num_genes, num_peaks). Peak-gene distance matrix, usually an exponential decay.
    gene_peak_fixed : torch.Tensor
        Shape (num_genes, num_peaks). A binary mask indicating allowable gene-peak links.
    insilico_act : torch.Tensor
        Shape (num_peaks, num_tfs). In silico ChIP-seq matrix for activators.
    insilico_rep : torch.Tensor
        Shape (num_peaks, num_tfs). In silico ChIP-seq matrix for repressors.
    phase : str, optional
        "warmup" or "grn". In "warmup", sets gene-peak and TF-binding matrices,
        and keeps them fixed or partially trainable. In "grn", sets the TF-binding
        matrices when given, replacing any copy from a checkpoint, and enables
        TF-gene parameters to be trainable.

    Returns
    -------
    None
        Modifies `model` in place, setting appropriate `.data` values and
        `.requires_grad` booleans.
    """
    with torch.no_grad():
        if phase != "grn":
            # 1) Set the fixed gene-peak mask.
            model.gene_peak_factor_fixed.data.copy_(gene_peak_fixed)
            model.gene_peak_factor_fixed.requires_grad = False

            # 2) Initialize the learnable gene-peak factor with distance-based weights.
            model.gene_peak_factor_learnt.data.copy_(gene_peak_distance_exp)
            model.gene_peak_factor_learnt.requires_grad = True

            # 3) Initialize TF binding matrices for activator & repressor.
            model.tf_binding_matrix_activator.data.copy_(insilico_act)
            model.tf_binding_matrix_activator.requires_grad = False

            model.tf_binding_matrix_repressor.data.copy_(insilico_rep)
            model.tf_binding_matrix_repressor.requires_grad = False

            model.tf_gene_topic_activator_grn.requires_grad = False
            model.tf_gene_topic_repressor_grn.requires_grad = False

        elif phase == "grn":
            # A checkpoint stores the binding it was saved with; phase 2 must use the current inputs.
            for name, binding in (("tf_binding_matrix_activator", insilico_act), ("tf_binding_matrix_repressor", insilico_rep)):
                if binding is None:
                    continue
                parameter = getattr(model, name)
                binding = torch.as_tensor(binding).to(parameter)
                if binding.shape != parameter.shape:
                    raise ValueError(f"{name}: inputs have shape {tuple(binding.shape)}, the model has {tuple(parameter.shape)}")
                if not torch.equal(parameter.data, binding):
                    logger.warning("%s differs from the loaded checkpoint; using the current inputs.", name)
                parameter.data.copy_(binding)
            model.gene_peak_factor_fixed.requires_grad = False
            model.tf_binding_matrix_activator.requires_grad = False
            model.tf_binding_matrix_repressor.requires_grad = False

            # Enable fine-tuning of TF-gene links per topic
            model.tf_gene_topic_activator_grn.requires_grad = True
            model.tf_gene_topic_repressor_grn.requires_grad = True

    print("scDORI parameters (peak-gene distance & TF binding) initialized and relevant parameters frozen.")
