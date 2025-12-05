"""Example usage of the Spagraph package.

Spagraph provides a complete pipeline for spatial transcriptomics analysis:
- Stage 1 (VAE): SC-ST integration and cell type clustering
- Stage 2 (GAT): Spatial deconvolution
- Stage 3 (CellCom): Cell-cell communication analysis
"""

from pathlib import Path

import spagraph


def main():
    """Run complete Spagraph pipeline"""
    
    # Define paths
    data_dir = Path("data")
    output_dir = Path("output")
    
    sc_file = str(data_dir / "sc_adata.h5ad")
    st_file = str(data_dir / "st_adata.h5ad")
    
    print("="*80)
    print("Spagraph Pipeline Example")
    print("="*80)
    
    # ========================================================================
    # Option 1: Run Stage 1 + Stage 2 separately
    # ========================================================================
    print("\n" + "="*80)
    print("Option 1: Run stages separately")
    print("="*80)
    
    # Stage 1: Train VAE
    stage1 = spagraph.vae(
        sc_file=sc_file,
        st_file=st_file,
        output_dir=str(output_dir / "stage1"),
        n_epochs=150,
        resolution=4.0,
    )
    print(f"Stage 1 complete: {stage1.n_clusters} clusters, {stage1.n_genes} genes")
    
    # Stage 2: Deconvolution (reuse Stage 1 artifacts)
    result = spagraph.deconv(
        st_file=st_file,
        vae=stage1,  # Pass Stage 1 artifacts directly
        stage2_epochs=200,
    )
    print(f"Stage 2 complete: Pearson={result['metrics']['pearson']:.4f}")
    print(f"Deconvolution matrix shape: {result['deconv'].shape}")
    
    # ========================================================================
    # Option 2: Run Stage 1 + Stage 2 in one call
    # ========================================================================
    print("\n" + "="*80)
    print("Option 2: Run deconvolution in one call")
    print("="*80)
    
    result = spagraph.deconv(
        sc_file=sc_file,
        st_file=st_file,
        output_dir=str(output_dir / "combined"),
        stage1_epochs=150,
        stage2_epochs=200,
        resolution=4.0,
    )
    print(f"Complete: {result['metrics']['n_clusters']} clusters")
    print(f"Pearson: {result['metrics']['pearson']:.4f}")
    print(f"MSE: {result['metrics']['mse']:.4f}")
    
    # Access deconvolution matrix directly
    deconv_df = result['deconv']
    print(f"\nDeconvolution matrix shape: {deconv_df.shape}")
    print(f"Cell types: {list(deconv_df.columns)}")
    print(f"\nAverage proportions:\n{deconv_df.mean().sort_values(ascending=False)}")
    
    # ========================================================================
    # Option 3: Run on multiple ST samples with same SC reference
    # ========================================================================
    print("\n" + "="*80)
    print("Option 3: Process multiple ST samples")
    print("="*80)
    
    # First, train VAE once
    stage1 = spagraph.vae(
        sc_file=sc_file,
        st_file=st_file,  # Use any ST file for initial training
        output_dir=str(output_dir / "shared_model"),
        n_epochs=150,
    )
    
    # Then apply to multiple ST samples
    st_samples = ["st_sample1.h5ad", "st_sample2.h5ad", "st_sample3.h5ad"]
    for st_sample in st_samples:
        st_path = str(data_dir / st_sample)
        result = spagraph.deconv(
            st_file=st_path,
            vae=stage1,  # Reuse the trained VAE
            output_dir=str(output_dir / Path(st_sample).stem),
            stage2_epochs=200,
        )
        print(f"{st_sample}: Pearson={result['metrics']['pearson']:.4f}")
    
    # ========================================================================
    # Stage 3: Cell Communication Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("Stage 3: Cell Communication Analysis")
    print("="*80)
    
    spagraph.cellcom(
        deconv_dir=str(output_dir / "combined"),  # Use deconvolution output
        st_h5ad=st_file,
        output_dir=str(output_dir / "cellcom"),
        epochs=100,
        batch_size=4,
    )
    print("Cell communication analysis complete!")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
