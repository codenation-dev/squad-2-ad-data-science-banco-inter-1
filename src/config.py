# algorithms parameters


class Configure:
    def __init__(self):
        self.pre_processing_params = {}
        self.feature_selection_params = {}
        self.model_params = {}
        self.evaluation_params = {}

    def set_pre_processing_params(self):
        d = {'threshold': 0.7,
             'EDA': ['fl_epp',
                     'qt_socios_pf',
                     'idade_maxima_socios',
                     'idade_minima_socios',
                     'qt_socios_st_regular',
                     'qt_socios_masculino',
                     'de_saude_rescencia',
                     'de_faixa_faturamento_estimado',
                     'de_faixa_faturamento_estimado_grupo',
                     'vl_faturamento_estimado_grupo_aux',
                     'idade_emp_cat',
                     'de_saude_rescencia',
                     'fl_me',
                     'fl_email',
                     'nu_meses_rescencia',
                     'fl_st_especial'
                     'sg_uf',
                     'sg_uf_matriz',
                     'nm_micro_regiao',
                     'nm_segmento',
                     'nm_divisao',
                     'de_natureza_juridica',
                     'setor']
             }
        self.pre_processing_params = d

