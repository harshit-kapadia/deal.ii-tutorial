
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

#include <cmath>



namespace LaplaceSolver
{

using namespace dealii;


template <int dim>
class Solution : public Function<dim>
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
  
  const Tensor<1,dim> x = p ;
  return_value += sin(2*M_PI*x.norm()) ;
  
  return return_value;
}

template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
  Tensor<1,dim> return_value;
  
  const Tensor<1,dim> x = p ;
  return_value = cos(2*M_PI*x.norm()) ;

  return return_value;
}

template <int dim>
class RightHandSide : public Function<dim>
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
  
  const Tensor<1,dim> x = p ;
      
  return_value += sin(2*M_PI*x.norm()) ;

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
  laplace(const FiniteElement<dim> &fe);
  ~laplace();
  void run ();

private:
  void make_grid ();
  void refine_grid () ;
  void setup_system ();
  void assemble_system ();
  void solve ();
  void output_results () ;

  Triangulation<dim> triangulation;
  SmartPointer<const FiniteElement<dim> > fe;
  DoFHandler<dim> dof_handler;

  SparsityPattern sparsity_pattern;
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
laplace<dim>::laplace (const FiniteElement<dim> &fe)  :  fe (&fe),  dof_handler (triangulation)
{}


template <int dim>
laplace<dim>::~laplace ()
{
  dof_handler.clear ();
}


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
  dof_handler.distribute_dofs (*fe);

  DynamicSparsityPattern dsp (dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);  

  system_matrix.reinit (sparsity_pattern);

  // exact_solution.reinit (dof_handler.n_dofs());
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

//  std::out << sizeof(exact_solution) << std::endl;
}


template <int dim>
void laplace<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(3);
  FEValues<dim> fe_values (*fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

   int   dofs_per_cell = fe->dofs_per_cell;
   int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const RightHandSide<dim> right_hand_side; // only quering data, never changing it; so declared constant
  std::vector<double>  rhs_values (n_q_points); // new
  const Solution<dim> exact_solution;

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
  typename DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      cell_matrix = 0;
      cell_rhs = 0;

      right_hand_side.value_list (fe_values.get_quadrature_points(), rhs_values); // new

      for (int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (int i=0; i<dofs_per_cell; ++i)
            for (int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) * fe_values.shape_grad (j, q_index) * fe_values.JxW (q_index));

          for (int i=0; i<dofs_per_cell; ++i)
            cell_rhs(i) += (fe_values.shape_value (i, q_index) * rhs_values [q_index] * fe_values.JxW (q_index));
        }
      
      cell->get_dof_indices (local_dof_indices);

      for (int i=0; i<dofs_per_cell; ++i)
        for (int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));

      for (int i=0; i<dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index,double> boundary_values;

  VectorTools::interpolate_boundary_values (dof_handler, 0, Solution<dim>(), boundary_values);

  MatrixTools::apply_boundary_values (boundary_values, system_matrix, solution, system_rhs);
}


template <int dim>
void laplace<dim>::solve ()
{
  SolverControl solver_control (5000, 1e-12);
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

  VectorTools::integrate_difference (dof_handler, solution, Solution<dim>(), difference_per_cell, QGauss<dim>(3), 
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
      // output_results ();

      process_solution (cycle);
    }


    convergence_table.set_precision("L2", 3);

    convergence_table.set_scientific("L2", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "$L^2$-error");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");


    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    std::string error_filename = "error";
    error_filename += ".tex";
    std::ofstream error_table_file(error_filename.c_str());
    convergence_table.write_tex(error_table_file);


    std::string error_txtfile = "error-txt";
    // error_filename += std::to_string(dim) ;
    error_filename += ".txt";
    std::ofstream error_txt_file(error_txtfile.c_str());
    convergence_table.write_text(error_txt_file);



    convergence_table.add_column_to_supercolumn("cycle", "n cells");
    convergence_table.add_column_to_supercolumn("cells", "n cells");
    std::vector<std::string> new_order;
    new_order.push_back("n cells");
    new_order.push_back("L2");
    convergence_table.set_column_order (new_order);

    convergence_table
    .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
    convergence_table
    .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);


    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    std::string conv_filename = "convergence";
    conv_filename += "-global";
    conv_filename += ".tex";
    std::ofstream table_file(conv_filename.c_str());
    convergence_table.write_tex(table_file);
}


}

int main ()
{
  const unsigned int dim = 2;
  const unsigned int poly_order = 1;  
  
  using namespace dealii;
  using namespace LaplaceSolver;

  // deallog.depth_console (2); // for CG iteration convergence info.

  FE_Q<dim> fe(poly_order);
  laplace<dim> problem (fe);
  problem.run ();

  return 0;
}
