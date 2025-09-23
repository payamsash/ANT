.. raw:: html

    <div style="text-align:center; margin-bottom:20px;">
        <img src="_static/ANT_Logo_Horizontal.svg" alt="ANT Logo" width="550">
    </div>

.. raw:: html

    <div style="height:30px;"></div>

Welcome to the **ANT** documentation! **ANT** is a open-source package which provides a comprehensive framework for *real-time M/EEG analysis and visualization*, including:

- *Real-time feature extraction* from M/EEG signals.
- Processing at multiple levels: *sensor-space to source-space*.
- *3D visualization of brain activation* on a rendered brain reconstructed from the subject's MRI.
- Integration with *visualization modules* to monitor *neurofeedback responses in real time*.
- *Real-time artifact correction methods*, including:
  - *ORICA* (Online Recursive ICA) for source separation.
  - *Adaptive regression (LMS)* for removing blink artifacts from EEG.
  - *Real-time SSP* (Signal Space Projection) for MEG artifact correction.
  - *HFC correction* for MEG high-frequency artifacts.
- *Lightweight management of experimental projects*, enabling organized and efficient experimentation.
- Flexible and extensible tools for *real-time neuroimaging, visualization, and neurofeedback applications*.

.. raw:: html

    <div style="height:30px;"></div>

.. raw:: html

    <!-- Top wide video (scaled 0.8) -->
    <div style="text-align:center; margin-bottom: 20px;">
        <video width="680" height="392" autoplay muted loop
               style="border-radius: 20px; border: 4px solid rgba(255,255,255,0.3);
                      box-shadow: 0 10px 20px rgba(0,0,0,0.5);">
            <source src="_static/nf_demo.mov" type="video/quicktime">
            Your browser does not support the video tag.
        </video>
    </div>

    <!-- Bottom row with two videos (scaled 0.8) -->
    <div style="display:flex; justify-content:center; gap:16px; margin-top:10px;">
        <video width="332" height="192" autoplay muted loop
               style="border-radius: 20px; border: 4px solid rgba(255,255,255,0.3);
                      box-shadow: 0 6px 15px rgba(0,0,0,0.4);">
            <source src="_static/brain.mov" type="video/quicktime">
            Your browser does not support the video tag.
        </video>

        <video width="332" height="192" autoplay muted loop
               style="border-radius: 20px; border: 4px solid rgba(255,255,255,0.3);
                      box-shadow: 0 6px 15px rgba(0,0,0,0.4);">
            <source src="_static/VisualTree.mp4" type="video/quicktime">
            Your browser does not support the video tag.
        </video>
    </div>


.. toctree::
   :hidden:

   API Reference <ant>
   Examples <auto_examples/index>

.. raw:: html

    <div style="height:30px;"></div>

.. raw:: html

    <h4 style="font-size:18px; margin-bottom:10px;">Installation</h4>

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


.. raw:: html

    <h4 style="font-size:18px; margin-bottom:10px;">Cite</h4>

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


.. raw:: html

    <h4 style="font-size:18px; margin-bottom:10px;">Supporting institutions</h4>

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
