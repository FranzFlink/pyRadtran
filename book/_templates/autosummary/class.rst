{{ fullname | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. automethod:: __init__

   .. autosummary::
      :toctree:
      :recursive:

      {% for item in methods %}
         ~{{ name }}.{{ item }}
      {%- endfor %}
      {% for item in attributes %}
         ~{{ name }}.{{ item }}
      {%- endfor %}