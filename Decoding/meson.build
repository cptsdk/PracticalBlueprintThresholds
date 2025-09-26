project('ffcc_blueprint', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3'])

m_dep = meson.get_compiler('cpp').find_library('m', required : true)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'cpp')
endif

lib = library('LossDec', ['CppCode/LossDec_funcs.cpp'])
lib = library('NoiseSampling', ['CppCode/NoiseSampling.cpp'])
