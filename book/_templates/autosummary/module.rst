{{ fullname | underline }}

.. automodule:: {{ fullname }}

.. autosummary::
   :toctree:
   :recursive:

   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}