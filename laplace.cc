
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


// template <int dim>
// class Solution : public Function<dim>
// {
// public:
//   Solution () : Function<dim>() {}
//   virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
//   virtual Tensor<1,dim> gradient (const Point<dim> &p, const unsigned int component = 0) const;
// };

// template <int dim>
// double Solution<dim>::value (const Point<dim> &p, const unsigned int) const
// {
//   double return_value = 0;
  
//   const Tensor<1,dim> x = p ;
//   return_value += ( (x[0]*x[0]*x[0] / 6) - (x[0] / 2) ) ;
  
//   return return_value;
// }

// template <int dim>
// Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &p, const unsigned int) const
// {
//   Tensor<1,dim> return_value;
  
//   const Tensor<1,dim> x = p ;
//   return_value[0] = (x[0]*x[0] / 2 ) - 0.5 ;

//   return return_value;
// }

// template <int dim>
// class RightHandSide : public Function<dim>
// {
// public:
//   RightHandSide () : Function<dim>() {}
//   virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
// };

// template <int dim>
// double RightHandSide<dim>::value (const Point<dim> &p, const unsigned int) const
// {
//   double return_value = 0;
  
//   const Tensor<1,dim> x = p ;
      
//   return_value += (-x[0]) ;

//   return return_value;
// }


template <int dim>
class Solution : public Function<dim>
{
public:
  Solution () : Function<dim>() {}
  virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
double Solution<dim>::value (const Point<dim> &p, const unsigned int) const
{
  double return_value = 0;
  
  const Tensor<1,dim> x = p ;
  return_value += ( (x[0]*x[0]) - (1.0/3) ) ;
  
  return return_value;
}

template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &p, const unsigned int) const
{
  Tensor<1,dim> return_value;
  
  const Tensor<1,dim> x = p ;
  return_value[0] = (2*x[0]) ;

  return return_value;
}

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide () : Function<dim>() {}
  virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p, const unsigned int) const
{
  double return_value = 0;
  
  const Tensor<1,dim> x = p ;
      
  return_value += (-2) ;

  return return_value;
}




// template <int dim>
// class SolutionBase
// {
// protected:
//   static const unsigned int n_source_centers = 3;
//   static const Point<dim>   source_centers[n_source_centers];
//   static const double       width;
// };

// template <>
// const Point<1>
// SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers]
//   = { Point<1>(-1.0 / 3.0),
//       Point<1>(0.0),
//       Point<1>(+1.0 / 3.0)
//     };

// template <>
// const Point<2>
// SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers]
//   = { Point<2>(-0.5, +0.5),
//       Point<2>(-0.5, -0.5),
//       Point<2>(+0.5, -0.5)
//     };

// template <int dim>
// const double SolutionBase<dim>::width = 1./8.;

// template <int dim>
// class Solution : public Function<dim>,
//   protected SolutionBase<dim>
// {
// public:
//   Solution () : Function<dim>() {}
//   virtual double value (const Point<dim>   &p,
//                         const unsigned int  component = 0) const;
//   virtual Tensor<1,dim> gradient (const Point<dim>   &p,
//                                   const unsigned int  component = 0) const;
// };

// template <int dim>
// double Solution<dim>::value (const Point<dim>   &p,
//                              const unsigned int) const
// {
//   double return_value = 0;
//   for (unsigned int i=0; i<this->n_source_centers; ++i)
//     {
//       const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
//       return_value += std::exp(-x_minus_xi.norm_square() /
//                                (this->width * this->width));
//     }
//   return return_value;
// }

// template <int dim>
// Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &p, const unsigned int) const
// {
//   Tensor<1,dim> return_value;
//   for (unsigned int i=0; i<this->n_source_centers; ++i)
//     {
//       const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
//       return_value += (-2 / (this->width * this->width) * std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) * x_minus_xi);
//     }
//   return return_value;
// }

// template <int dim>
// class RightHandSide : public Function<dim>,
//   protected SolutionBase<dim>
// {
// public:
//   RightHandSide () : Function<dim>() {}
//   virtual double value (const Point<dim>   &p,
//                         const unsigned int  component = 0) const;
// };

// template <int dim>
// double RightHandSide<dim>::value (const Point<dim>   &p,
//                                   const unsigned int) const
// {
//   double return_value = 0;
//   for (unsigned int i=0; i<this->n_source_centers; ++i)
//     {
//       const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
//       return_value += ((2*dim - 4*x_minus_xi.norm_square()/
//                         (this->width * this->width)) /
//                        (this->width * this->width) *
//                        std::exp(-x_minus_xi.norm_square() /
//                                 (this->width * this->width)));
//       // return_value += std::exp(-x_minus_xi.norm_square() /
//       //                          (this->width * this->width));
//     }
//   return return_value;
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
  void initial_condition () ;
  void assemble_system ();
  void solve_current_time (double dt) ;
  void solve ();
  double check_convergence () ;
  void output_results () ;

  Triangulation<dim> triangulation;
  SmartPointer<const FiniteElement<dim> > fe;
  DoFHandler<dim> dof_handler;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;
  SparseMatrix<double> basis_inner_product ;

  void process_solution (const unsigned int cycle);
  Vector<double> solution;
  Vector<double> system_rhs;

  ConvergenceTable convergence_table;
  // TableHandler output_table ;
  // DataOut<dim> data_out ;
  ConvergenceTable plot_table ;
};


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
  triangulation.refine_global (2);
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
  basis_inner_product.reinit (sparsity_pattern) ;

  // exact_solution.reinit (dof_handler.n_dofs());
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

//  std::out << sizeof(exact_solution) << std::endl;
}


template <int dim>
void laplace<dim>::initial_condition ()
{
  const bool 	omit_zeroing_entries = false ;
  solution.reinit (dof_handler.n_dofs(), omit_zeroing_entries); 

  // VectorTools::interpolate(	dof_handler, Solution<dim>(), solution )	;
}


template <int dim>
void laplace<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(3);
  QGauss<dim-1> face_quadrature_formula(3);

  FEValues<dim> fe_values (*fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula, update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);

   int   dofs_per_cell = fe->dofs_per_cell;
   int   n_q_points    = quadrature_formula.size();
   const unsigned int n_face_q_points = face_quadrature_formula.size();


  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  FullMatrix<double> cell_basis_inner_product (dofs_per_cell, dofs_per_cell) ;

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
      cell_basis_inner_product = 0;

      right_hand_side.value_list (fe_values.get_quadrature_points(), rhs_values); // new

      for (int q_index=0; q_index<n_q_points; ++q_index)
      {
        for (int i=0; i<dofs_per_cell; ++i)
        {
          for (int j=0; j<dofs_per_cell; ++j)
          {
            cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) * fe_values.shape_grad (j, q_index) * fe_values.JxW (q_index)) ;
            cell_basis_inner_product(i,j) += (fe_values.shape_value (i, q_index) * fe_values.shape_value (j, q_index) * fe_values.JxW (q_index)) ;
          }
        }
        for (int i=0; i<dofs_per_cell; ++i)
          cell_rhs(i) += (fe_values.shape_value (i, q_index) * rhs_values [q_index] * fe_values.JxW (q_index));
      }

      for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
      {
        if (cell->face(face_number)->at_boundary()) // check whether current face of the cell is at boundary
        { 
          std::cout << "\n face center: " << cell->face(face_number)->center() << std::endl ;
          fe_face_values.reinit (cell, face_number); // compute the values of the shape functions and the other quantities
          for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point) // loop over all quad pts
            {
              const double neumann_value = (exact_solution.gradient(fe_face_values.quadrature_point(q_point)) * fe_face_values.normal_vector(q_point));
              std::cout << "  neumann value: " << neumann_value 
                        << "  Gradient: " << exact_solution.gradient(fe_face_values.quadrature_point(q_point)) 
                        << "  normal_vector on face: " << fe_face_values.normal_vector(q_point) 
                        << std::endl ;
              for (int i=0; i<dofs_per_cell; ++i)
              { 
                std::cout << "\n  cell rhs before boundary contribution: " << cell_rhs(i) << std::endl ;
                cell_rhs(i) += ( neumann_value * fe_face_values.shape_value(i,q_point) * fe_face_values.JxW(q_point) );
                std::cout << "  dof: " << i 
                          << "  cell rhs: " << cell_rhs(i) 
                          << "  shape value: " << fe_face_values.shape_value(i,q_point) 
                          << "  JxW: " << fe_face_values.JxW(q_point) 
                          << std::endl ;
              }            
            } 
        }
      }

      
      cell->get_dof_indices (local_dof_indices);

      for (int i=0; i<dofs_per_cell; ++i)
      {
        for (int j=0; j<dofs_per_cell; ++j)
        {
          system_matrix.add (local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j)) ;
          basis_inner_product.add (local_dof_indices[i], local_dof_indices[j], cell_basis_inner_product(i,j)) ;
        }
      }

      for (int i=0; i<dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  // std::map<types::global_dof_index,double> boundary_values;
  // VectorTools::interpolate_boundary_values (dof_handler, 0, Solution<dim>(), boundary_values);
  // MatrixTools::apply_boundary_values (boundary_values, system_matrix, solution, system_rhs);
}


template <int dim>
void laplace<dim>::solve_current_time (double dt)
{
  SparseMatrix<double> local_system_matrix ;
  local_system_matrix.reinit(sparsity_pattern) ;
  local_system_matrix.copy_from(system_matrix) ;
  
  SparseMatrix<double> local_basis_inner_product ;
  local_basis_inner_product.reinit(sparsity_pattern) ;
  local_basis_inner_product.copy_from(basis_inner_product) ;

  Vector<double> local_system_rhs ;  
  local_system_rhs = (system_rhs) ;
  
  int dofs_per_cell = fe->dofs_per_cell;

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
  typename DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  
  local_system_matrix *= (-dt) ;
  // for (; cell!=endc; ++cell)
  // {
  //   cell->get_dof_indices (local_dof_indices);
  //   for (int i=0; i<dofs_per_cell; ++i){
  //     for (int j=0; j<dofs_per_cell; ++j) {
  //       local_system_matrix.add (local_dof_indices[i], local_dof_indices[j], local_basis_inner_product(local_dof_indices[i], local_dof_indices[j])) ;
  //     }
  //   }
  // }
	for(unsigned int i=0; i<dof_handler.n_dofs(); ++i){
		for(unsigned int j=0; j<dof_handler.n_dofs(); ++j){
			local_system_matrix.add (i, j, local_basis_inner_product(i, j)) ;
		}
	}


  Vector<double> temporary(dof_handler.n_dofs()) ;

  local_system_rhs *= (dt) ;
  local_system_matrix.vmult(temporary, solution) ;
  local_system_rhs += temporary ; 


  // std::map<types::global_dof_index,double> boundary_values;
  // VectorTools::interpolate_boundary_values (dof_handler, 0, Solution<dim>(), boundary_values);
  // MatrixTools::apply_boundary_values (boundary_values, local_basis_inner_product, solution, local_system_rhs);

  // for (std::map<types::global_dof_index,double>::iterator it=boundary_values.begin(); it!=boundary_values.end(); ++it)
  //   std::cout << it->first << " => " << it->second << '\n';


  SolverControl solver_control (5000, 1e-12);
  SolverCG<> solver (solver_control);
  solver.solve (local_basis_inner_product, solution, local_system_rhs, PreconditionIdentity());  
}


template <int dim>
void laplace<dim>::solve ()
{
  SolverControl solver_control (5000, 1e-12);
  SolverCG<> solver (solver_control);

  solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity());
}

template <int dim>
double laplace<dim>::check_convergence ()
{
  Vector<double> rhs_from_solution(dof_handler.n_dofs()) ;
  
  system_matrix.vmult(rhs_from_solution, solution) ;
  rhs_from_solution -= system_rhs ;

  return rhs_from_solution.l2_norm() ; 
}


template <int dim>
void laplace<dim>::output_results ()
{
  // std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
  // std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;


  // DataOut<dim> data_out ;
  // data_out.attach_dof_handler (dof_handler);
  // data_out.add_data_vector (solution, "solution");
  // data_out.build_patches ();
  // std::ofstream plot_file("plot.txt");
  // data_out.write_gnuplot(plot_file);

  // output_table.add_value("solution", solution[2]);
  // std::ofstream plot_file("plot.txt");
  // output_table.write_text(plot_file);

  // plot_table.add_value("solution", solution[5]);
  // std::ofstream plot_file("plot.txt");
  // plot_table.write_text(plot_file);
}


template <int dim>
void laplace<dim>::process_solution (const unsigned int cycle)
{
  Vector<float> difference_per_cell (triangulation.n_active_cells());

  VectorTools::integrate_difference (dof_handler, solution, Solution<dim>(), difference_per_cell, QGauss<dim>(3), VectorTools::L2_norm);



  // std::ofstream outdata; // outdata is like cin
  //  int i; // loop index

  // outdata.open("plot.txt"); // opens the file
  //  if( !outdata ) { // file couldn't be opened
  //     std::cerr << "Error: file could not be opened" << std::endl;
  //     exit(1);
  //  }

  // for (int i=0; i<triangulation.n_active_cells(); ++i)
  //     outdata << difference_per_cell(i) << std::endl;
   
  // outdata.close();



  const double L2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);

  const unsigned int n_active_cells=triangulation.n_active_cells();
  const unsigned int n_dofs=dof_handler.n_dofs();
  // std::cout << "Cycle " << cycle << ':'
  //           << std::endl
  //           << "   Number of active cells:       "
  //           << n_active_cells
  //           << std::endl
  //           << "   Number of degrees of freedom: "
  //           << n_dofs
  //           << std::endl;
  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);


  std::ofstream outdata; // outdata is like cin
  outdata.open("data.txt"); // opens the file
   if( !outdata ) { // file couldn't be opened
      std::cerr << "Error: file could not be opened" << std::endl;
      exit(1);
   }

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
  typename DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();

  Solution<dim> exact_solution;

  for (; cell!=endc; ++cell)
  {
    Point<dim> center = cell->center();
    Vector<double> solution_value(1);
    double exact_solution_value;
    double error_value;

    VectorTools::point_value	(dof_handler,
                                    solution,
                                    center,
                                    solution_value);

    	
    exact_solution_value = exact_solution.value(center);
    error_value = fabs(solution_value(0)-exact_solution_value);

    outdata<< center(0) << " "<< solution_value(0)<< " " << exact_solution_value << " " << error_value << std::endl;
  }
    
  outdata.close();
}


template <int dim>
void laplace<dim>::run ()
{
  const unsigned int n_cycles = 5;
  double dt = 0.0001 ;
  for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
    {
      double convg_norm = 1, T = 0.0 ;

      std::cout << "\n\n\n\n Current cycle: " << cycle << std::endl ;      

      if (cycle == 0)
        make_grid ();
      else
        refine_grid ();

      setup_system ();
      initial_condition ();

      assemble_system ();
        // printf(" Basis inner product- matrix (column by column) : ") ;
        // for (unsigned int i=0; i<dof_handler.n_dofs(); ++i){
        //   for (unsigned int j=0; j<dof_handler.n_dofs(); ++j) {
        //     printf("\n \t %0.4f ", basis_inner_product(j,i)) ;
        //   }
        //   printf("\n") ;
        // }   
      
      // convg_norm = check_convergence () ; 
      // std::cout << "   ||Ax-f||: " << convg_norm << std::endl ;
        // Vector<double> rhs_from_solution(dof_handler.n_dofs()) ;
        // system_matrix.vmult(rhs_from_solution, solution) ;
        // std::cout << "norm: " << rhs_from_solution.linfty_norm() << std::endl ;
      while(convg_norm > 1e-08)
      {
        T += dt ;
        std::cout << "\n\n\n  Current time: " << T << std::endl ;
        
        solve_current_time (dt); 
                
        convg_norm = check_convergence () ;
        std::cout << "    norm: " << convg_norm << std::endl ;

        // for(unsigned int i=0; i<dof_handler.n_dofs(); ++i)
        //   printf("\n      %0.6f ", solution(i)) ;
      }

      // solve();

      process_solution (cycle);
    }
    output_results() ;

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
  const unsigned int dim = 1;
  const unsigned int poly_degree = 1;  
  
  using namespace dealii;
  using namespace LaplaceSolver;

  // deallog.depth_console (2); // for CG iteration convergence info.

  FE_Q<dim> fe(poly_degree);
  laplace<dim> problem (fe);
  problem.run ();

  return 0;
}
