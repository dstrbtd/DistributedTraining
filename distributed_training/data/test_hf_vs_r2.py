import asyncio
from transformers import AutoTokenizer
import bittensor as bt

from tplr.dataset import DatasetLoader
from tplr.r2_dataset import R2DatasetLoader

def hf_index_from_r2(r2_index: int) -> int:
    """Convert R2 index to HF dataset index"""
    row_groups_crossed = r2_index // 1000
    return r2_index + row_groups_crossed

class ComparisonTest:
    def __init__(self):
        self.uid = 42
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.batch_size = 2
        self.sequence_length = 32
        self.n_pages = 1
        self.offset = 0

    async def get_r2_pages(self):
        """Get pages using R2DatasetLoader only"""
        pages = await R2DatasetLoader.next_pages(
            offset=self.offset,
            n_pages=self.n_pages,
            seed=self.uid
        )
        
        pages = [('CC-MAIN-2014-52', 2001, 'train')]
        print('pages', pages)
        return pages

    async def run_hf_dataloader(self, pages):
        """Run the HF dataloader with converted indices"""
        converted_pages = [
            (dataset_id, hf_index_from_r2(index), split)
            for dataset_id, index, split in pages
        ]
        
        print('hf converted pages', converted_pages)
        
        dataset = await DatasetLoader.create(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            pages_info=converted_pages,
            tokenizer=self.tokenizer,
        )
        
        batch, labels = next(iter(dataset))
        return batch.numpy(), labels.numpy()

    async def run_r2_dataloader(self, pages):
        """Run the R2 dataloader with original pages"""
        print('r2 pages', pages)
        
        loader = await R2DatasetLoader.create(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            pages_info=pages,
            tokenizer=self.tokenizer
        )
        
        for i, batch in enumerate(loader):
            if i == 0:
                return batch, batch[:, 1:]
            break

    async def compare_outputs(self):
        """Run both dataloaders and compare their raw outputs"""
        pages = await self.get_r2_pages()
        
        hf_batch, hf_labels = await self.run_hf_dataloader(pages)
        r2_batch, r2_labels = await self.run_r2_dataloader(pages)
        
        print("\n=== HF Dataloader Output ===")
        print(f"Batch shape: {hf_batch.shape}")
        print(f"First 5 tokens: {hf_batch[0, :5].tolist()}")
        
        print("\n=== R2 Dataloader Output ===")
        print(f"Batch shape: {r2_batch.shape}")
        print(f"First 5 tokens: {r2_batch[0, :5].tolist()}")
        
        assert hf_batch.shape == r2_batch.shape, f"Batch shape mismatch: {hf_batch.shape} vs {r2_batch.shape}"
        
        assert (hf_batch == r2_batch).all(), "Batch content differs"
        
        print("\n✅ Both dataloaders produce identical output!")

async def main():
    tester = ComparisonTest()
    await tester.compare_outputs()

if __name__ == "__main__":
    asyncio.run(main())