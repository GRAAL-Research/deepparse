.. list-table::
        :header-rows: 1

        *   - Model
            - Hidden size
            - Test accuracy of the released model (%)
            - Test accuracy, mean ± std over the seeds (%)
        *   - ``model2vec-32m``
            - 512
            - 98.13
            - 97.75 ± 0.56 (5 seeds)
        *   - ``model2vec-32m-attention``
            - 1024
            - 98.42
            - 97.72 ± 0.72 (3 seeds)
        *   - ``model2vec-8m``
            - 1024
            - 97.95
            - 97.78 ± 0.32 (5 seeds)
        *   - ``model2vec-8m-attention``
            - 256
            - 98.17
            - 97.99 ± 0.29 (5 seeds)
        *   - ``model2vec-multilingual``
            - 1024
            - 97.65
            - 97.55 ± 0.66 (4 seeds)
        *   - ``model2vec-multilingual-attention``
            - 512
            - 98.87
            - 97.49 ± 0.83 (5 seeds)
