
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>



namespace LaplaceSolver
{

using namespace dealii;



template <int dim>
class SolutionBase
{
protected:
  static const unsigned int n_source_centers = 3;
  static const Point<dim>   source_centers[n_source_centers];
  static const double       width;
};

template <>
const Point<1>
SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers]
  = { Point<1>(-1.0 / 3.0),
      Point<1>(0.0),
      Point<1>(+1.0 / 3.0)
    };

template <>
const Point<2>
SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers]
  = { Point<2>(-0.5, +0.5),
      Point<2>(-0.5, -0.5),
      Point<2>(+0.5, -0.5)
    };

template <int dim>
const double SolutionBase<dim>::width = 1./8.;

template <int dim>
class Solution : public Function<dim>,
  protected SolutionBase<dim>
{
public:
  Solution () : Function<dim>() {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
};

template <int dim>
double Solution<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
{
  double return_value = 0;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
      const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
      return_value += std::exp(-x_minus_xi.norm_square() /
                               (this->width * this->width));
    }
  return return_value;
}

template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
  Tensor<1,dim> return_value;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
      const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
      return_value += (-2 / (this->width * this->width) *
                       std::exp(-x_minus_xi.norm_square() /
                                (this->width * this->width)) *
                       x_minus_xi);
    }
  return return_value;
}

template <int dim>
class RightHandSide : public Function<dim>,
  protected SolutionBase<dim>
{
public:
  RightHandSide () : Function<dim>() {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim>   &p,
                                  const unsigned int) const
{
  double return_value = 0;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
      const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
      return_value += ((2*dim - 4*x_minus_xi.norm_square()/
                        (this->width * this->width)) /
                       (this->width * this->width) *
                       std::exp(-x_minus_xi.norm_square() /
                                (this->width * this->width)));
      return_value += std::exp(-x_minus_xi.norm_square() /
                               (this->width * this->width));
    }
  return return_value;
}







// template<int dim>
// class exact_solution:public Function<dim>
// {
// public:
// 	exact_solution...
// 	virtual double value()..
// };

// constructor of exact solution
// template<dim> double exact_solution<dim>::value()

// {

// 	Return exact solution

// }


template<int dim>
class laplace
{
public:
  laplace();
  void run ();

private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void output_results () ;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  SparseMatrix<double> system_matrix;

  void process_solution (const unsigned int cycle);
  Vector<double> solution;
  Vector<double> system_rhs;

  ConvergenceTable convergence_table;
};

// laplace:error_evaluation()
// 		{
// 		exact_solution<2> exact_poission;
// 		integrate_difference();


// 		return(final l2 norm);
// 		}

template <int dim>
laplace<dim>::laplace ()  :  fe (2),  dof_handler (triangulation)
{}


template <int dim>  
void laplace<dim>::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (3);
}


template <int dim>
void laplace<dim>::refine_grid ()
{
	triangulation.refine_global (1);
}


template <int dim>
void laplace<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  // DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
  // DoFTools::make_sparsity_pattern (dof_handler, dsp);
  // // hanging_node_constraints.condense (dsp);
  // sparsity_pattern.copy_from (dsp);
  DynamicSparsityPattern pattern(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, pattern);

  system_matrix.reinit (pattern);

  // exact_solution.reinit (dof_handler.n_dofs());
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

//  std::out << sizeof(exact_solution) << std::endl;
}


template <int dim>
void laplace<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(3);
  FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values);

   int   dofs_per_cell = fe.dofs_per_cell;
   int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const RightHandSide<dim> right_hand_side;
  // std::vector<double>  rhs_values (n_q_points);
  const Solution<dim> exact_solution;

  DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
  DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      cell_matrix = 0;
      cell_rhs = 0;

      for (int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (int i=0; i<dofs_per_cell; ++i)
            for (int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) * fe_values.shape_grad (j, q_index) * fe_values.JxW (q_index));

          for (int i=0; i<dofs_per_cell; ++i)
            cell_rhs(i) += (fe_values.shape_value (i, q_index) * 1 * fe_values.JxW (q_index));
        }
      cell->get_dof_indices (local_dof_indices);

      for (int i=0; i<dofs_per_cell; ++i)
        for (int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));

      for (int i=0; i<dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index,double> boundary_values;

  VectorTools::interpolate_boundary_values (dof_handler, 0, ZeroFunction<2>(), boundary_values);

  MatrixTools::apply_boundary_values (boundary_values, system_matrix, solution, system_rhs);
}


template <int dim>
void laplace<dim>::solve ()
{
  SolverControl solver_control (1000, 1e-12);
  SolverCG<> solver (solver_control);

  solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity());
}


template <int dim>
void laplace<dim>::output_results ()
{
  std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
}


template <int dim>
void laplace<dim>::process_solution (const unsigned int cycle)
{
  Vector<float> difference_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (dof_handler,
                                     solution,
                                     Solution<dim>(),
                                     difference_per_cell,
                                     QGauss<dim>(3),
                                     VectorTools::L2_norm);
  const double L2_error = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::L2_norm);

  const unsigned int n_active_cells=triangulation.n_active_cells();
  const unsigned int n_dofs=dof_handler.n_dofs();
  std::cout << "Cycle " << cycle << ':'
            << std::endl
            << "   Number of active cells:       "
            << n_active_cells
            << std::endl
            << "   Number of degrees of freedom: "
            << n_dofs
            << std::endl;
  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);

}


template <int dim>
void laplace<dim>::run ()
{
  const unsigned int n_cycles = 5;
  for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
    {
      if (cycle == 0)
        make_grid ();
      else
        refine_grid ();

  setup_system ();
  assemble_system ();
  solve ();
  output_results ();

  process_solution (cycle);
}


}

int main ()
{
  using namespace dealii;
  using namespace LaplaceSolver;

  deallog.depth_console (2); // for CG iteration convergence info.
  laplace<2> problem1;
  problem1.run ();
  return 0;
}
