Advancing Neurofeedback in Tinnitus
=================

Welcome to the ANT documentation! ANT is an open-source package which provides tools for real-time processing and visualization of M/EEG
neurofeedback experiments.

.. raw:: html

    <div style="text-align:center; margin: 0 0;">
        <video width="850" height="490" autoplay muted 
                style="border-radius: 20px; border: 4px solid #333;">
            <source src="_static/nf_demo.mov" type="video/quicktime">
            Your browser does not support the video tag.
        </video>
    </div>


.. toctree::
   :hidden:

   ant
   auto_examples/index

Installation
~~~~~~~~~~~~~~

You can install ANT via `PyPI <https://pypi.org>`_, `Conda <https://conda-forge.org>`_, or
directly from `Source <https://github.com/payamsash/ANT>`_.

💻 **PyPI**

.. code-block:: console

    $ pip install ant

🌟 **Conda**

.. code-block:: console

    $ conda install -c conda-forge ant

📦 **From Source (GitHub)**

.. code-block:: console

    $ pip install git+https://github.com/payamsash/ANT


Cite
~~~~~~~~~~~~~~

If you use ``ANT``, please consider citing our paper :footcite:`shabestari2025advances`.

.. footbibliography::

.. tab-set::

    .. tab-item:: APA

        .. code-block:: none

            Shabestari, P. S., Ribes, D., Défayes, L., Cai, D., Groves, E., Behjat, H. H., ... & Neff, P. (2025, June). Advances on Real Time M/EEG Neural Feature Extraction. In 2025 IEEE 38th International Symposium on Computer-Based Medical Systems (CBMS) (pp. 337-338). IEEE.

    .. tab-item:: BibTeX

        .. code-block:: bibtex

            @inproceedings{shabestari2025advances,
               title={Advances on Real Time M/EEG Neural Feature Extraction},
               author={Shabestari, Payam S and Ribes, Delphine and D{\'e}fayes, Lara and Cai, Danpeng and Groves, Emily and Behjat, Harry H and Van de Ville, Dimitri and Kleinjung, Tobias and Naas, Adrian and Henchoz, Nicolas and others},
               booktitle={2025 IEEE 38th International Symposium on Computer-Based Medical Systems (CBMS)},
               pages={337--338},
               year={2025},
               organization={IEEE}
            }


Supporting institutions
~~~~~~~~~~~~~~

.. image:: _static/SNF.png
    :align: right
    :alt: SNSF
    :width: 400

The development of ``ANT`` was supported by the
`Swiss National Science Foundation <https://www.snf.ch/en>`_.

.. toctree::
    :hidden:

    resources/install.rst
    api/index.rst
    resources/command_line.rst
    resources/implementations.rst
    generated/tutorials/index.rst
    generated/examples/index.rst



For more information, see:

- GitHub repository: https://github.com/payamsash/ANT
- Tutorials and guides in the `examples <auto_examples/index.html>`_ section.
