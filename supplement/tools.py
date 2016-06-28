from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from patsy import dmatrices


def read_results(f_name='results_all.csv'):
    all_results = pd.read_csv(f_name, low_memory=False)

    columns = [
        c for c in all_results.columns
        if c not in ('fold_label', 'men', 'SimLex999', 'KS14', 'GS11', 'PhraseRel', 'GS12')
    ]

    all_results.loc[:, 'fold_label'].fillna('max', inplace=True)
    all_results.loc[:, 'neg'].fillna('N/A', inplace=True)
    for c in columns:
        all_results.loc[:, c].fillna('', inplace=True)

    all_results = all_results.set_index(columns + ['fold_label']).unstack('fold_label')

    results = all_results.groupby(level=[c for c in columns if c != 'fold_num']).mean()
    results = results.reorder_levels([1, 0], axis=1)

    return results


def max_(*others):
    def max_(df, dataset):
        best_index = df['max', dataset].argmax()
        best_result = df.loc[best_index]
        result = pd.concat(
            [
                pd.Series({d: best_result['max', d] for d in (dataset, *others)}),
                pd.Series(best_index, index=df.index.names),
            ]
        )

        return result

    return max_


def cross_validation(*others):
    def cross_validation(df, dataset):
        best_training_index = df['training', dataset].argmax()
        best_training_setting = df.loc[best_training_index]

        result = pd.concat(
            [
                pd.Series(
                    {
                        'training': df['training', dataset],
                        'testing': best_training_setting['testing', dataset],
                        **{d: best_training_setting['max', d] for d in (dataset, *others)}
                    }
                ),
                pd.Series(best_training_index, index=df.index.names),
            ]
        )

        return result

    return cross_validation


def _selection_plot(results, selection_type, dataset):
    for hue in (
            'freq',
            'neg',
            'cds',
            'similarity',
            'discr',
    ):
        g = sns.factorplot(
            data=results,
            y=dataset,
            x='dimensionality',
            hue=hue,
            hue_order={
                'cds': ('global', '1', '0.75'),
                'neg': (0.2, 0.5, 0.7, 1, 1.4, 2, 5, 'N/A'),
                'similarity': ('cos', 'correlation', 'inner_product'),
                'freq': ('1', 'n', 'logn'),
                'discr': ('pmi', 'cpmi', 'spmi', 'scpmi'),
            }[hue],
            col='operator',
            col_order=('head', 'add', 'mult', 'kron') if dataset in ('KS14', 'GS11', 'GS12', 'PhraseRel') else ['head'],
            size=3,
            aspect=1.6,
        )

        g.fig.savefig('figures/{}-{}-selection-{}.pdf'.format(dataset, selection_type, hue))


def plot_selection(results, dataset, selector_function, plot=True):
    if not isinstance(selector_function, str):
        selection = (
            results
            .groupby(level=['operator', 'dimensionality'])
            .apply(selector_function, dataset=dataset)
        )
        selection_type = selection['selection'] = selector_function.__name__

    else:
        selection = results
        selection_type = selector_function

    if plot:
        _selection_plot(selection, selection_type=selection_type, dataset=dataset)

    if not isinstance(selector_function, str):
        return selection


def plot_interaction(data, hue, dataset_name):
    g = sns.factorplot(
        data=data.reset_index(),
        y=dataset_name,
        x='dimensionality',
        hue=hue,
        hue_order={
            'cds': ('global', '1', '0.75'),
            'neg': (
                0.2,
                0.5,
                0.7,
                1,
                1.4,
                2,
                5,
                7,
                'N/A',
            ),
            'similarity': ('cos', 'correlation', 'inner_product'),
            'freq': ('1', 'n', 'logn'),
            'discr': ('pmi', 'cpmi', 'spmi', 'scpmi'),
        }[hue],
        size=3,
        aspect=1.6,
        sharey=False,
        dodge=0.3,
        col='operator',
        col_order=('head', 'add', 'mult', 'kron') if dataset_name in ('KS14', 'GS11') else ['head'],
    )

    g.fig.savefig('figures/{}-interaction-{}.pdf'.format(dataset_name, hue))


def plot_parameter_selection_comparison(results, original_dataset, other_dataset=None, ax=None, operator=None):
    other_dataset = other_dataset or original_dataset

    results = pd.concat(results)

    if operator is not None:
        results = results.loc[operator]

    g = sns.pointplot(
        data=results,
        y=other_dataset,
        x='dimensionality',
        # col='operator',
        hue='selection',
        hue_order=('max_', 'cross_validation', 'heuristics') + (('upper bound',) if other_dataset != original_dataset else tuple()),
        dodge=0.3,
        ax=ax,
    )
    # g.fig.savefig('figures/{}-parameter_selection_comparison.pdf'.format(dataset))


def anova(response, predictors, data, interaction_only=None):
    data = data['max'][response].reset_index().replace('', np.nan)

    if interaction_only is None:
        formula = (
            '{response} ~ {predictors} + '.format(
                response=response,
                predictors=' + '.join(predictors)
            ) +
            ' + '.join('{} * {}'.format(*param) for param in combinations(predictors, 2))
        )
    else:
        formula = (
            '{response} ~ '.format(response=response) +
            ' + '.join(' * '.join(param) for param in combinations(predictors, interaction_only))
        )

    y, X = dmatrices(
        formula,
        data=data,
        return_type='dataframe',
    )

    mod = sm.OLS(y, X)
    res = mod.fit()

    return res


def ablation(response, predictors, data, extra=tuple()):
    full_rsquared_adj = anova(
        response=response,
        predictors=predictors,
        data=data,
    ).rsquared_adj

    for left_out in predictors:
        rest = set(predictors).difference([left_out])

        partial_rsquared_adj = anova(
            response=response,
            predictors=rest,
            data=data
        ).rsquared_adj

        yield (response, left_out, full_rsquared_adj - partial_rsquared_adj) + tuple(extra)


def calculate_feature_ablation(responses, predictors, data):
    def ablations():
        for response in responses:
            yield pd.DataFrame(
                ablation(
                    response=response,
                    predictors=predictors,
                    data=data,
                ),
                columns=('response', 'predictor', 'partial R2'),
            ).set_index(['response', 'predictor'])

    df = pd.concat(ablations()).unstack('response')
    df.sort_values(df.columns[0], inplace=True)

    return df


def average_error(max_selection, heuristics_selection, dataset):
    return np.round(
        (
            (
                max_selection[dataset] -
                heuristics_selection[dataset]
            ) /
            max_selection[dataset]
        ).dropna().mean(),
        3
    )
