function L = estimate(e, model, x_train, y_train, x_test, y_test, varargin)
%
% ESTIMATE - evaluate splitsample estimate of a performance criterion
%
%    L = ESTIMATE(CROSSVALIDATION,MODEL,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST)
%    estimates the split sample estimate of a performance criterion, for 
%    training data (X_TRAIN,Y_TRAIN) and test data (X_TEST,Y_TEST).

%
% File        : @splitsample/estimate.m
%
% Date        : Sunday 26th August 2007
%
% Author      : Dr Gavin C. Cawley
%
% Description : Evaluate a split-sample performance estimate for a generalised
%               kernel machine.
%
% History     : 26/08/2007 - v1.00
%
% Copyright   : (c) Dr Gavin C. Cawley, November 2003.
%
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program; if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
%

criterion = get(e, 'criterion'); 
model     = train(model, x_train, y_train, varargin{:});
[mu,eta]  = fwd(model, x_test);
L         = evaluate(criterion, model, y_test, mu, eta);

% bye bye...

